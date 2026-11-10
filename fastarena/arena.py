"""A stand-in for Visual-fighting's blender/arena.py that draws with OpenGL instead of Blender.

Same names and pools (Arena.balls, held, pick, fx, ...), same helpers (world, set_glow, _bsdf),
so Visual-fighting's own blender/render_fight.py runs on it unchanged (see launch.py). The
stage, meshes and materials come from assets/arena.npz/.json, exported from the real Blender
arena once (tools/export_arena.py); the cast's colours and pictures are set here.

The look, as in the Blender version: Blender-style area and spot lights, soft shadows from the
key light, a mirror pass for the glossy floor, glowing (emission) materials, see-through effects,
depth of field, bloom, a slight lens fringe, and AgX with the Punchy look.
"""
from __future__ import annotations

import ctypes, json, math, sys, types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vmath import Vector, Euler, euler_matrix  # noqa: E402

W, H = 1080, 1920
PX = 0.01
RING_R = 4.70
BALL_R = 0.64
WALL_H = 0.42
SENSOR = 0.036              # vertical sensor height (m), as the Blender camera


def gentle():
    if sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)


def lin(c):
    def f(v):
        v = v / 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return (f(c[0]), f(c[1]), f(c[2]), 1.0)


def world(x, y, z=0.0):
    return Vector(((x - 540) * PX, -(y - 1130) * PX, z))


# ---------------------------------------------------------------- stand-ins for Blender data
class Mat:
    def __init__(self, name, d):
        self.name = name
        self.kind = d["kind"]
        self.blended = d.get("blended", False)
        self.base = list(d.get("base", [0.8, 0.8, 0.8]))
        self.rough = d.get("rough", 0.5); self.metal = d.get("metal", 0.0)
        self.coat = d.get("coat", 0.0); self.coat_rough = d.get("coat_rough", 0.03)
        self.spec = d.get("spec", 0.5)
        self.emit = list(d.get("emit", [0, 0, 0])); self.strength = d.get("strength", 0.0)
        self.alpha = d.get("alpha", 1.0)
        self.pic = None


class _Sock:
    def __init__(self, m, attr): self.m, self.attr = m, attr
    default_value = property(lambda s: getattr(s.m, s.attr), lambda s, v: setattr(s.m, s.attr, float(v)))


class _BSDF:
    NAMES = {"Emission Strength": "strength", "Alpha": "alpha", "Roughness": "rough", "Metallic": "metal"}
    def __init__(self, m): self.inputs = {k: _Sock(m, a) for k, a in self.NAMES.items()}


def _bsdf(mat):
    return _BSDF(mat)


def set_glow(m, strength=None, alpha=None, col=None):
    if strength is not None: m.strength = float(strength)
    if alpha is not None: m.alpha = max(0.0, min(1.0, float(alpha)))
    if col is not None: m.emit = list(lin(col)[:3])


class _Slot:
    def __init__(self, m): self.material = m


class Obj:
    def __init__(self, name, mesh, mats, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), hide=False):
        self.name = name
        self.data = types.SimpleNamespace(name=mesh)
        self.material_slots = [_Slot(m) for m in mats]
        self.location, self.rotation_euler, self.scale, self.hide_render = loc, rot, scale, hide

    location = property(lambda s: s._loc, lambda s, v: setattr(s, "_loc", Vector(v)))

    def matrix(self):
        M = np.eye(4)
        M[:3, :3] = euler_matrix(self.rotation_euler) * np.asarray(tuple(self.scale), float)[None, :]
        M[:3, 3] = tuple(self._loc)
        return M


class Camera(Obj):
    def __init__(self):
        super().__init__("cam", "cam", [])
        self.data = types.SimpleNamespace(lens=52.0, sensor_height=36.0, clip_start=0.1, clip_end=200.0,
                                          dof=types.SimpleNamespace(use_dof=True, aperture_fstop=5.6,
                                                                    focus_distance=10.0))
        self.R = np.eye(3)

    def view(self):
        V = np.eye(4); V[:3, :3] = self.R.T; V[:3, 3] = -self.R.T @ np.array(tuple(self._loc))
        return V

    def half_height(self):
        return (self.data.sensor_height / 2) / self.data.lens


def world_to_camera_view(scene, cam, co):
    """Blender's bpy_extras.object_utils.world_to_camera_view: (x, y) in 0..1 of the frame, z = depth."""
    p = cam.R.T @ (np.array(tuple(co), float) - np.array(tuple(cam.location)))
    z = -p[2]
    th = cam.half_height(); tw = th * W / H
    if z == 0: return Vector((0.5, 0.5, 0.0))
    return Vector(((p[0] / (z * tw) + 1) / 2, (p[1] / (z * th) + 1) / 2, z))


# ---------------------------------------------------------------- shaders
MAIN_VS = """
#version 330
uniform mat4 u_vp; uniform mat4 u_model; uniform mat3 u_nmat;
in vec3 in_pos; in vec3 in_nrm;
out vec3 v_wpos; out vec3 v_nrm; out vec3 v_lpos;
void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_wpos = w.xyz; v_nrm = u_nmat * in_nrm; v_lpos = in_pos;
    gl_Position = u_vp * w;
}
"""
MAIN_FS = """
#version 330
#define PI 3.14159265
#define NL 5
in vec3 v_wpos; in vec3 v_nrm; in vec3 v_lpos;
out vec4 f_col;
uniform int u_kind;                  // 0 basic, 1 glow, 2 ball, 3 floor
uniform vec3 u_base; uniform float u_rough, u_metal, u_coat, u_coat_rough, u_spec;
uniform vec3 u_emit; uniform float u_strength, u_alpha;
uniform sampler2D u_pic; uniform float u_pic_k;
uniform vec3 u_campos; uniform vec3 u_ambient;
uniform vec3 L_pos[NL]; uniform vec3 L_dir[NL]; uniform vec3 L_col[NL];
uniform float L_cos[NL]; uniform float L_blend[NL]; uniform float L_rad[NL]; uniform float L_spec[NL];
uniform sampler2D u_shadow; uniform mat4 u_svp; uniform float u_srange; uniform float u_ssize;
uniform sampler2D u_refl; uniform vec2 u_res; uniform int u_reflpass;

float D_GGX(float NoH, float a) { float a2 = a * a; float d = NoH * NoH * (a2 - 1.0) + 1.0; return a2 / (PI * d * d); }
float V_SJ(float NoV, float NoL, float a) {
    float a2 = a * a;
    return 0.5 / (NoL * sqrt(NoV * NoV * (1.0 - a2) + a2) + NoV * sqrt(NoL * NoL * (1.0 - a2) + a2) + 1e-5);
}
vec3 F_S(vec3 f0, float c) { return f0 + (1.0 - f0) * pow(1.0 - c, 5.0); }
float F_Sf(float f0, float c) { return f0 + (1.0 - f0) * pow(1.0 - c, 5.0); }
vec3 srgb2lin(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }

const vec2 DISK[16] = vec2[](vec2(-0.94, -0.40), vec2(0.95, -0.77), vec2(-0.09, -0.93), vec2(0.34, 0.29),
    vec2(-0.92, 0.46), vec2(-0.82, -0.15), vec2(-0.38, 0.28), vec2(-0.37, -0.40), vec2(0.20, 0.79),
    vec2(0.44, -0.90), vec2(0.96, 0.27), vec2(-0.25, 0.97), vec2(0.54, -0.21), vec2(0.17, -0.34),
    vec2(0.74, 0.62), vec2(-0.61, 0.79));

float shadow(vec3 p, vec3 n) {
    vec4 s = u_svp * vec4(p + n * 0.03, 1.0);
    vec3 c = s.xyz * 0.5 + 0.5;
    if (c.x <= 0.0 || c.x >= 1.0 || c.y <= 0.0 || c.y >= 1.0) return 1.0;
    float dr = c.z * u_srange;                       // distance from the light (light-space)
    float search = 0.35 / u_ssize;                   // search radius (uv), ~35 cm
    float sum = 0.0; float cnt = 0.0;
    for (int i = 0; i < 16; i++) {
        float db = texture(u_shadow, c.xy + DISK[i] * search).r * u_srange;
        if (db < dr - 0.02) { sum += db; cnt += 1.0; }
    }
    if (cnt == 0.0) return 1.0;
    float db = sum / cnt;
    float pen = clamp(5.0 * (dr - db) / db, 0.02, 0.6);   // the key light is a 5 m disc
    float r = 0.5 * pen / u_ssize;
    float lit = 0.0;
    for (int i = 0; i < 16; i++) {
        float d = texture(u_shadow, c.xy + DISK[i] * r).r * u_srange;
        lit += (d < dr - 0.02) ? 0.0 : 1.0;
    }
    return lit / 16.0;
}

vec3 lobe(vec3 N, vec3 V, vec3 L, float a, vec3 f0, float rad, float d) {
    float a2 = clamp(a + rad / (2.0 * d), 0.002, 1.0);
    vec3 Hh = normalize(V + L);
    float NoL = max(dot(N, L), 0.0), NoV = max(dot(N, V), 1e-4), NoH = max(dot(N, Hh), 0.0);
    float norm = (a / a2) * (a / a2);
    return F_S(f0, max(dot(V, Hh), 0.0)) * D_GGX(NoH, a2) * V_SJ(NoV, NoL, a2) * norm;
}

void main() {
    if (u_reflpass == 1 && v_wpos.z < -0.001) discard;
    if (u_kind == 1) { f_col = vec4(u_emit * u_strength * u_alpha, u_alpha); return; }
    vec3 N = normalize(v_nrm); if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(u_campos - v_wpos);
    vec3 base = u_base; vec3 emit = u_emit * u_strength;
    float floor_line = 0.0;
    if (u_kind == 2) {                        // the character printed on the front of the ball
        vec2 uv = v_lpos.xz * u_pic_k + 0.5;
        vec4 t = texture(u_pic, uv);
        float inside = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
        float a = t.a * inside * step(v_lpos.y, 0.0);
        base = mix(base, srgb2lin(t.rgb), a);
    } else if (u_kind == 3) {                 // floor: wave rings + spherical gradient (object coords)
        vec3 o = v_wpos + vec3(0.0, 0.0, 0.15);
        float fac = 0.5 - 0.5 * cos(length(o) * 0.55 * 20.0);
        float ramp = clamp((fac - 0.955) / 0.03, 0.0, 1.0);
        float grad = max(0.999999 - length(o) / 4.7, 0.0);
        base = mix(vec3(0.001821, 0.002428, 0.006995), vec3(0.010330, 0.015996, 0.076185), grad);
        emit = u_emit * ramp * grad * 0.45;
    }
    float NoV = max(dot(N, V), 1e-4);
    float ar = max(u_rough * u_rough, 0.002), ac = max(u_coat_rough * u_coat_rough, 0.002);
    vec3 f0 = mix(vec3(0.08 * u_spec), base, u_metal);
    float fc = F_Sf(0.04, NoV) * u_coat;
    vec3 diff = vec3(0.0), spec = vec3(0.0);
    for (int i = 0; i < NL; i++) {
        vec3 Lv = L_pos[i] - v_wpos; float d = length(Lv); vec3 L = Lv / d;
        float ce = dot(L_dir[i], -L);
        float att = L_cos[i] < -1.5 ? max(ce, 0.0)
                  : smoothstep(0.0, 1.0, (ce - L_cos[i]) / L_blend[i]);
        if (att <= 0.0) continue;
        float NoL = max(dot(N, L), 0.0);
        if (NoL <= 0.0) continue;
        vec3 E = L_col[i] * att / (d * d) * NoL;
        if (i == 0) E *= shadow(v_wpos, N);
        diff += E;
        if (L_spec[i] > 0.0) {
            vec3 s = lobe(N, V, L, ar, f0, L_rad[i], d) * (1.0 - fc)
                   + lobe(N, V, L, ac, vec3(0.04), L_rad[i], d) * u_coat;
            spec += E * s * L_spec[i];
        }
    }
    vec3 col = (diff + u_ambient * PI) * base * (1.0 - u_metal) / PI * (1.0 - fc) + spec + emit;
    if (u_kind == 3 && u_reflpass == 0) {     // the glossy floor mirrors the scene (planar reflection)
        vec2 suv = gl_FragCoord.xy / u_res;
        vec2 px = 4.0 / u_res;                // a small blur, so the half-size mirror image doesn't look blocky
        vec3 sharp = (textureLod(u_refl, suv, 1.5).rgb * 2.0
                    + textureLod(u_refl, suv + vec2(px.x, px.y), 2.0).rgb + textureLod(u_refl, suv + vec2(-px.x, px.y), 2.0).rgb
                    + textureLod(u_refl, suv + vec2(px.x, -px.y), 2.0).rgb + textureLod(u_refl, suv - px, 2.0).rgb) / 6.0;
        vec3 soft = textureLod(u_refl, suv, 4.5).rgb;
        col += sharp * fc + soft * F_Sf(0.04, NoV) * (1.0 - fc) * 0.8;
    }
    f_col = vec4(col * u_alpha, u_alpha);
}
"""
DEPTH_VS = """
#version 330
uniform mat4 u_vp; uniform mat4 u_model;
in vec3 in_pos;
void main() { gl_Position = u_vp * u_model * vec4(in_pos, 1.0); }
"""
DEPTH_FS = "#version 330\nvoid main() {}\n"

QUAD_VS = """
#version 330
in vec2 in_xy; out vec2 uv;
void main() { uv = in_xy * 0.5 + 0.5; gl_Position = vec4(in_xy, 0.0, 1.0); }
"""
RESOLVE_FS = """
#version 330
uniform sampler2DMS u_col; uniform sampler2DMS u_dep;
uniform float u_near, u_far, u_focus, u_cocK;    // CoC (px, radius) = u_cocK * |z - focus| / z
in vec2 uv; out vec4 f;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 c = vec3(0.0); float z = 0.0;
    for (int s = 0; s < 4; s++) {
        c += texelFetch(u_col, p, s).rgb;
        float d = texelFetch(u_dep, p, s).r * 2.0 - 1.0;
        z += 2.0 * u_near * u_far / (u_far + u_near - d * (u_far - u_near));
    }
    c *= 0.25; z *= 0.25;
    f = vec4(c, u_cocK * (z - u_focus) / z);       // + behind the focus, - in front
}
"""
DOF_FS = """
#version 330
uniform sampler2D u_img; uniform vec2 u_px; uniform float u_maxr;
in vec2 uv; out vec4 f;
void main() {
    vec4 c0 = texture(u_img, uv);
    float r0 = abs(c0.a);
    vec3 acc = c0.rgb / max(r0 * r0, 1.0); float wsum = 1.0 / max(r0 * r0, 1.0);
    for (int i = 1; i < 40; i++) {
        float t = sqrt(float(i) / 40.0); float a = float(i) * 2.39996;
        vec2 o = vec2(cos(a), sin(a)) * t * u_maxr;
        vec4 s = texture(u_img, uv + o * u_px);
        float rs = abs(s.a);
        if (s.a > 0.0) rs = min(rs, max(c0.a, 0.0) + 1.0);   // a blurred background can't cover what is in front
        float w = clamp(rs - length(o) + 1.0, 0.0, 1.0) / max(rs * rs, 1.0);
        acc += s.rgb * w; wsum += w;
    }
    f = vec4(acc / wsum, c0.a);
}
"""
BRIGHT_FS = """
#version 330
uniform sampler2D u_img; in vec2 uv; out vec4 f;
void main() {
    vec3 c = texture(u_img, uv).rgb;
    float m = max(c.r, max(c.g, c.b));
    f = vec4(c * max(m - 1.0, 0.0) / max(m, 1e-4), 1.0);
}
"""
DOWN_FS = """
#version 330
uniform sampler2D u_img; uniform vec2 u_px; in vec2 uv; out vec4 f;
void main() {
    vec3 c = texture(u_img, uv).rgb * 4.0;
    c += texture(u_img, uv + u_px * vec2(-1, -1)).rgb + texture(u_img, uv + u_px * vec2(1, -1)).rgb;
    c += texture(u_img, uv + u_px * vec2(-1, 1)).rgb + texture(u_img, uv + u_px * vec2(1, 1)).rgb;
    f = vec4(c / 8.0, 1.0);
}
"""
UP_FS = """
#version 330
uniform sampler2D u_img; uniform vec2 u_px; in vec2 uv; out vec4 f;
void main() {
    vec3 c = texture(u_img, uv).rgb * 4.0;
    c += (texture(u_img, uv + u_px * vec2(-1, 0)).rgb + texture(u_img, uv + u_px * vec2(1, 0)).rgb
        + texture(u_img, uv + u_px * vec2(0, -1)).rgb + texture(u_img, uv + u_px * vec2(0, 1)).rgb) * 2.0;
    c += texture(u_img, uv + u_px * vec2(-1, -1)).rgb + texture(u_img, uv + u_px * vec2(1, -1)).rgb;
    c += texture(u_img, uv + u_px * vec2(-1, 1)).rgb + texture(u_img, uv + u_px * vec2(1, 1)).rgb;
    f = vec4(c / 16.0, 1.0);
}
"""
FINAL_FS = """
#version 330
uniform sampler2D u_img; uniform sampler2D u_bloom; uniform float u_bloom_k;
uniform float u_dist, u_disp; uniform float u_aspect;
in vec2 uv; out vec4 f;
vec3 agx_curve(vec3 x) {
    vec3 x2 = x * x; vec3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
vec3 agx_punchy(vec3 v) {
    const mat3 m = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                        0.0784335999999992, 0.878468636469772, 0.0784336,
                        0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const mat3 mi = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                         -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                         -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    const float lo = -12.47393, hi = 4.026069;
    v = m * max(v, vec3(1e-10));
    v = clamp(log2(v), lo, hi); v = (v - lo) / (hi - lo);
    v = agx_curve(v);
    float luma = dot(v, vec3(0.2126, 0.7152, 0.0722));
    v = pow(max(v, vec3(0.0)), vec3(1.35));
    v = luma + 1.4 * (v - luma);
    return clamp(mi * v, 0.0, 1.0);
}
vec3 scene(vec2 p) { return texture(u_img, p).rgb + u_bloom_k * texture(u_bloom, p).rgb; }
void main() {
    vec2 d = (uv - 0.5) * vec2(u_aspect, 1.0);
    float r2 = dot(d, d);
    vec3 c;
    c.r = scene(0.5 + d * (1.0 + (u_dist + u_disp) * r2) / vec2(u_aspect, 1.0)).r;
    c.g = scene(0.5 + d * (1.0 + u_dist * r2) / vec2(u_aspect, 1.0)).g;
    c.b = scene(0.5 + d * (1.0 + (u_dist - u_disp) * r2) / vec2(u_aspect, 1.0)).b;
    f = vec4(agx_punchy(c), 1.0);
}
"""

gl = lambda M: np.ascontiguousarray(np.asarray(M, "f4").T).tobytes()


def perspective(half_h, aspect, near, far):
    P = np.zeros((4, 4))
    P[0, 0] = 1 / (half_h * aspect); P[1, 1] = 1 / half_h
    P[2, 2] = (far + near) / (near - far); P[2, 3] = 2 * far * near / (near - far); P[3, 2] = -1
    return P


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    M = np.eye(4); M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def U(prog, name, v):
    """Set a uniform if the shader uses it (unused ones are compiled away)."""
    if name in prog:
        u = prog[name]
        if isinstance(v, bytes): u.write(v)
        else: u.value = v


# ---------------------------------------------------------------- the arena
class Arena:
    def __init__(self, cast, scale=100, threads=4, samples=16, pic_dir: Path | None = None):
        self.w, self.h = W * scale // 100, H * scale // 100
        self.cast = cast
        self.cam = Camera()
        self.sc = types.SimpleNamespace(render=types.SimpleNamespace(filepath="", resolution_x=W, resolution_y=H,
                                                                     resolution_percentage=scale),
                                        camera=self.cam)
        info = json.loads((HERE / "arena.json").read_text())
        arr = np.load(HERE / "arena.npz")
        self.mats = {k: Mat(k, d) for k, d in info["materials"].items()}
        self.objs = {}
        for d in info["objects"]:
            self.objs[d["name"]] = Obj(d["name"], d["mesh"], [self.mats[m] for m in d["mats"]], d["loc"],
                                       tuple(d["rot"]), tuple(d["scale"]), d["hide"])
        self.world_col = np.array(info["world"]["col"]) * info["world"]["strength"]
        self._recolour(pic_dir)
        self._pools(info["pools"])
        self._gpu(arr, info)
        self.save_pool = ThreadPoolExecutor(max_workers=3); self.pending = []
        sys.modules["bpy"].ops.render.render = lambda **k: self.render(self.sc.render.filepath)

    # -- the cast: colours and pictures
    def _recolour(self, pic_dir):
        self.pics = {}
        for i, p in enumerate(self.cast):
            c = list(lin(p["col"])[:3])
            b = self.mats[f"ball{i}"]; b.base = c; b.pic = p["pic"]
            img = Image.open(Path(pic_dir) / f"{p['pic']}.png").convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
            self.pics[p["pic"]] = img
            for m in (f"aura{i}", f"under{i}"): self.mats[m].emit = c
            self.mats[f"shard{i}"].base = c; self.mats[f"shard{i}"].emit = c

    def _pools(self, P):
        o, m = self.objs.__getitem__, self.mats.__getitem__
        self.balls = [dict(obj=o(b["obj"]), mat=m(b["mat"]), ice=o(b["ice"]), aura=o(b["aura"]),
                           aura_m=m(b["aura_m"]), under=o(b["under"]), under_m=m(b["under_m"])) for b in P["balls"]]
        for k in ("held", "pick", "drop"):
            setattr(self, k, [{w: o(n) for w, n in d.items()} for d in P[k]])
        for k in ("bombs", "pads", "danger", "bullets", "rockets", "sights"):
            setattr(self, k, [o(n) for n in P[k]])
        self.shards = [[o(n) for n in s] for s in P["shards"]]
        self.fx = {k: [(o(a), m(b)) for a, b in pool] for k, pool in P["fx"].items()}
        self.bullet_m = {k: m(n) for k, n in P["bullet_m"].items()}
        for k in ("danger_m", "pad_m", "red_glow", "neon_m", "neon2_m"):
            setattr(self, k, m(P[k]))
        for b in self.bullets:                       # each bullet has its own material slot
            b.material_slots = [_Slot(b.material_slots[0].material)]
        dyn = set()
        for b in self.balls: dyn |= {b[k].name for k in ("obj", "ice", "aura", "under")}
        for k in ("held", "pick", "drop"):
            for d in getattr(self, k): dyn |= {x.name for x in d.values()}
        for k in ("bombs", "pads", "danger", "bullets", "rockets", "sights"):
            dyn |= {x.name for x in getattr(self, k)}
        for s in self.shards: dyn |= {x.name for x in s}
        for pool in self.fx.values(): dyn |= {a.name for a, _ in pool}
        self.dynamic = [self.objs[n] for n in dyn]
        self.static = [x for n, x in self.objs.items() if n not in dyn]

    # -- GPU setup
    def _gpu(self, arr, info):
        ctx = self.ctx = moderngl.create_standalone_context(require=330)
        self.p_main = ctx.program(vertex_shader=MAIN_VS, fragment_shader=MAIN_FS)
        self.p_depth = ctx.program(vertex_shader=DEPTH_VS, fragment_shader=DEPTH_FS)
        post = lambda fs: ctx.program(vertex_shader=QUAD_VS, fragment_shader=fs)
        self.p_resolve, self.p_dof, self.p_bright = post(RESOLVE_FS), post(DOF_FS), post(BRIGHT_FS)
        self.p_down, self.p_up, self.p_final = post(DOWN_FS), post(UP_FS), post(FINAL_FS)
        quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4"))
        self.quads = {p: ctx.vertex_array(p, [(quad, "2f", "in_xy")]) for p in
                      (self.p_resolve, self.p_dof, self.p_bright, self.p_down, self.p_up, self.p_final)}

        def vaos(pos, nrm):
            vbo = ctx.buffer(np.hstack([pos, nrm]).astype("f4"))
            return (ctx.vertex_array(self.p_main, [(vbo, "3f 3f", "in_pos", "in_nrm")]),
                    ctx.vertex_array(self.p_depth, [(vbo, "3f 12x", "in_pos")]))
        # meshes split by material slot
        self.parts = {}
        for name in {o.data.name for o in self.dynamic}:
            pos, nrm, mi = arr[name + ":pos"], arr[name + ":nrm"], arr[name + ":mi"]
            for k in np.unique(mi):
                sel = np.repeat(mi == k, 3)
                self.parts.setdefault(name, []).append((int(k), *vaos(pos[sel], nrm[sel])))
        # the stage never moves: bake it into world space, one batch per material
        batches = {}
        for o in self.static:
            M = o.matrix(); Nm = np.linalg.inv(M[:3, :3]).T
            pos, nrm, mi = arr[o.data.name + ":pos"], arr[o.data.name + ":nrm"], arr[o.data.name + ":mi"]
            for k in np.unique(mi):
                sel = np.repeat(mi == k, 3)
                p = pos[sel] @ M[:3, :3].T + M[:3, 3]; n = nrm[sel] @ Nm.T
                n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
                m = o.material_slots[k].material
                batches.setdefault(m.name, (m, [], [], o.name == "floor"))
                batches[m.name][1].append(p); batches[m.name][2].append(n)
        self.stage = [(m, *vaos(np.vstack(ps), np.vstack(ns)), is_floor) for m, ps, ns, is_floor in batches.values()]
        # lights (all aimed at the centre of the ring in arena.py)
        L = sorted(info["lights"], key=lambda l: l["name"] != "key")       # the key light first: it casts the shadow
        lp = np.array([l["loc"] for l in L], float)
        ld = -lp / np.linalg.norm(lp, axis=1, keepdims=True)
        lc = np.array([np.array(l["col"]) * l["energy"] / (4 * math.pi) for l in L])
        lcos = [math.cos(l["spot_size"] / 2) if l["type"] == "SPOT" else -2.0 for l in L]
        lbl = [max(1e-4, (1 - c) * l.get("spot_blend", 0)) if l["type"] == "SPOT" else 1.0 for c, l in zip(lcos, L)]
        lrad = [l.get("size", 0.5) / 2 if l["type"] == "AREA" else 0.25 for l in L]
        pm = self.p_main
        U(pm, "L_pos", lp.astype("f4").tobytes()); U(pm, "L_dir", ld.astype("f4").tobytes())
        U(pm, "L_col", lc.astype("f4").tobytes()); U(pm, "L_cos", np.array(lcos, "f4").tobytes())
        U(pm, "L_blend", np.array(lbl, "f4").tobytes()); U(pm, "L_rad", np.array(lrad, "f4").tobytes())
        U(pm, "L_spec", np.array([l["spec"] for l in L], "f4").tobytes())
        U(pm, "u_ambient", tuple(self.world_col))
        # the key light's shadow map (orthographic, covering the ring)
        self.S_EXT, S_RANGE = 6.5, 30.0
        key = lp[0]
        self.svp = np.array([[1 / self.S_EXT, 0, 0, 0], [0, 1 / self.S_EXT, 0, 0],
                             [0, 0, -2 / S_RANGE, -1], [0, 0, 0, 1]]) @ look_at(key, (0, 0, 0), (0, 1, 0))
        U(pm, "u_svp", gl(self.svp)); U(pm, "u_srange", S_RANGE); U(pm, "u_ssize", 2 * self.S_EXT)
        self.shadow_tex = ctx.depth_texture((2048, 2048))
        self.shadow_tex.compare_func = ""
        self.shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)
        # targets
        w, h = self.w, self.h
        self.ms_col = ctx.texture((w, h), 4, samples=4, dtype="f2")
        self.ms_dep = ctx.depth_texture((w, h), samples=4)
        self.ms_fbo = ctx.framebuffer(color_attachments=[self.ms_col], depth_attachment=self.ms_dep)
        rw, rh = w // 2, h // 2
        self.refl_tex = ctx.texture((rw, rh), 4, dtype="f2")
        self.refl_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.refl_tex.repeat_x = self.refl_tex.repeat_y = False
        self.refl_fbo = ctx.framebuffer(color_attachments=[self.refl_tex], depth_attachment=ctx.depth_renderbuffer((rw, rh)))
        def tex(size):
            t = ctx.texture(size, 4, dtype="f2"); t.repeat_x = t.repeat_y = False
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            return t, ctx.framebuffer(color_attachments=[t])
        self.res_tex, self.res_fbo = tex((w, h))
        self.dof_tex, self.dof_fbo = tex((w, h))
        self.bloom = [tex((max(1, w >> k), max(1, h >> k))) for k in range(1, 8)]
        self.bloom_up = [tex((max(1, w >> k), max(1, h >> k))) for k in range(1, 7)]
        self.out_fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 3)])
        self.pic_tex = {}
        for code, img in self.pics.items():
            t = ctx.texture(img.size, 4, img.tobytes())
            t.build_mipmaps(); t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False; t.anisotropy = 8.0
            self.pic_tex[code] = t

    # -- the camera (same as arena.py)
    def aim_camera(self, pos, target, lens=None, fstop=None, roll=0.0):
        self.cam.location = pos
        d = Vector(target) - Vector(pos)
        q = d.to_track_quat("-Z", "Y")
        if roll:
            from vmath import Quaternion
            q = q @ Quaternion((0, 0, 1), roll)
        self.cam.rotation_euler = q.to_euler()
        self.cam.R = euler_matrix(self.cam.rotation_euler)
        self.cam.data.dof.focus_distance = d.length
        if lens: self.cam.data.lens = lens
        if fstop: self.cam.data.dof.aperture_fstop = fstop

    def instance(self, tpl, name):
        raise NotImplementedError("the pools are made at export time")

    # -- drawing
    def _use_mat(self, m):
        p = self.p_main
        kind = {"basic": 0, "glow": 1, "ball": 2, "floor": 3}[m.kind]
        U(p, "u_kind", kind); U(p, "u_base", tuple(m.base)); U(p, "u_rough", m.rough); U(p, "u_metal", m.metal)
        U(p, "u_coat", m.coat); U(p, "u_coat_rough", m.coat_rough); U(p, "u_spec", m.spec)
        U(p, "u_emit", tuple(m.emit)); U(p, "u_strength", m.strength); U(p, "u_alpha", m.alpha)
        if kind == 2:
            self.pic_tex[m.pic].use(2); U(p, "u_pic", 2); U(p, "u_pic_k", 1 / (BALL_R * 2 * 0.80))

    @staticmethod
    def _see_through(m):
        return m.blended and (m.alpha < 0.999 or m.kind != "glow")

    def _items(self):
        """(object matrix, material, main vao, depth vao) for every visible moving thing."""
        out = []
        for o in self.dynamic:
            if o.hide_render: continue
            M = o.matrix()
            if abs(np.linalg.det(M[:3, :3])) < 1e-12: continue      # zero size: nothing to draw
            for k, vm, vd in self.parts[o.data.name]:
                m = o.material_slots[k].material
                if m.alpha <= 0.002 and m.blended: continue
                out.append((M, m, vm, vd, o))
        return out

    def _pass(self, fbo, vp, campos, items, reflect):
        p = self.p_main
        U(p, "u_vp", gl(vp)); U(p, "u_campos", tuple(campos)); U(p, "u_reflpass", 1 if reflect else 0)
        I3 = gl(np.eye(4)); N3 = gl(np.eye(3))
        opaque, clear = [], []
        for m, vm, vd, is_floor in self.stage:
            if reflect and is_floor: continue
            (clear if self._see_through(m) else opaque).append((None, m, vm))
        for M, m, vm, vd, o in items:
            (clear if self._see_through(m) else opaque).append((M, m, vm))
        def draw(lst):
            for M, m, vm in lst:
                if M is None:
                    U(p, "u_model", I3); U(p, "u_nmat", N3)
                else:
                    U(p, "u_model", gl(M)); U(p, "u_nmat", gl(np.linalg.inv(M[:3, :3]).T))
                self._use_mat(m)
                vm.render()
        ctx = self.ctx
        ctx.disable(moderngl.BLEND); ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.CULL_FACE)
        draw(opaque)
        cp = np.array(tuple(campos))
        clear.sort(key=lambda it: -np.linalg.norm((it[0][:3, 3] if it[0] is not None else np.zeros(3)) - cp))
        ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        fbo.depth_mask = False
        draw(clear)
        fbo.depth_mask = True
        ctx.disable(moderngl.BLEND)

    def _post(self, p, fbo, tex_units, **unis):
        fbo.use()
        for k, (t, unit) in tex_units.items():
            t.use(unit); U(p, k, unit)
        for k, v in unis.items(): U(p, k, v)
        self.quads[p].render(moderngl.TRIANGLES)

    def draw(self):
        ctx, cam = self.ctx, self.cam
        items = self._items()
        d = cam.data
        near, far = d.clip_start, d.clip_end
        P = perspective(cam.half_height(), self.w / self.h, near, far)
        V = cam.view()
        campos = np.array(tuple(cam.location))
        # 1. shadow map
        self.shadow_fbo.use(); self.shadow_fbo.clear(depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        pd = self.p_depth; U(pd, "u_vp", gl(self.svp))
        U(pd, "u_model", gl(np.eye(4)))
        for m, vm, vd, is_floor in self.stage:
            if m.kind != "glow": vd.render()
        for M, m, vm, vd, o in items:
            if m.kind != "glow" and not m.blended:
                U(pd, "u_model", gl(M)); vd.render()
        self.shadow_tex.use(1); U(self.p_main, "u_shadow", 1)
        # 2. the floor's mirror image (half size)
        mirror = np.diag([1.0, 1.0, -1.0, 1.0])
        self.refl_fbo.use(); self.refl_fbo.clear(*self.world_col, 1.0, depth=1.0)
        U(self.p_main, "u_res", (float(self.w), float(self.h)))
        self._pass(self.refl_fbo, P @ V @ mirror, campos * np.array([1, 1, -1]), items, True)
        self.refl_tex.build_mipmaps()
        # 3. the main picture (4x multisampled, half floats)
        self.ms_fbo.use(); self.ms_fbo.clear(*self.world_col, 1.0, depth=1.0)
        self.refl_tex.use(3); U(self.p_main, "u_refl", 3)
        self._pass(self.ms_fbo, P @ V, campos, items, False)
        ctx.disable(moderngl.DEPTH_TEST)
        # 4. resolve + circle of confusion, depth of field
        L = d.lens / 1000; N = d.dof.aperture_fstop; fd = max(d.dof.focus_distance, L * 2)
        cocK = 0.5 * (L / N) * L / (fd - L) / SENSOR * self.h          # radius in px
        self._post(self.p_resolve, self.res_fbo, {"u_col": (self.ms_col, 0), "u_dep": (self.ms_dep, 1)},
                   u_near=near, u_far=far, u_focus=fd, u_cocK=cocK)
        img = self.res_tex
        maxr = min(cocK, 14.0 * self.h / 1920)
        if maxr > 0.6:
            self._post(self.p_dof, self.dof_fbo, {"u_img": (self.res_tex, 0)}, u_px=(1 / self.w, 1 / self.h), u_maxr=maxr)
            img = self.dof_tex
        # 5. bloom: bright parts, blurred down a chain of smaller images and back up
        self._post(self.p_bright, self.bloom[0][1], {"u_img": (img, 0)})
        for k in range(1, len(self.bloom)):
            src = self.bloom[k - 1][0]
            self._post(self.p_down, self.bloom[k][1], {"u_img": (src, 0)}, u_px=(1 / src.width, 1 / src.height))
        up = self.bloom[-1][0]
        ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE
        for k in range(len(self.bloom) - 2, -1, -1):
            t, fbo = self.bloom_up[k]
            fbo.use(); fbo.clear()
            ctx.disable(moderngl.BLEND)
            self._post(self.p_up, fbo, {"u_img": (up, 0)}, u_px=(1 / up.width, 1 / up.height))
            ctx.enable(moderngl.BLEND)
            self._post(self.p_up, fbo, {"u_img": (self.bloom[k][0], 0)}, u_px=(1 / t.width, 1 / t.height))
            up = t
        ctx.disable(moderngl.BLEND)
        # 6. lens fringe + AgX Punchy -> 8-bit
        self._post(self.p_final, self.out_fbo, {"u_img": (img, 0), "u_bloom": (up, 1)},
                   u_bloom_k=0.13, u_dist=-0.01, u_disp=0.012, u_aspect=self.w / self.h)

    def render(self, path):
        self.draw()
        raw = self.out_fbo.read(components=3, alignment=1)
        w, h = self.w, self.h
        def save():
            Image.frombytes("RGB", (w, h), raw).transpose(Image.FLIP_TOP_BOTTOM).save(path, compress_level=1)
        self.pending.append(self.save_pool.submit(save))
        while len(self.pending) > 6:
            self.pending.pop(0).result()
