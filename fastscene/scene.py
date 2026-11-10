"""A drop-in replacement for Science_vid/blender/scene.py (farm theme), drawn with OpenGL.

Same class and calls as the Blender version (Scene, set_cells, set_tufts, set_floor,
set_walker, set_camera, render), so Science_vid's own blender/render_video.py runs on it
unchanged. Each frame is one shadow pass + one main pass on the GPU, then a JPEG written
in the background.

Farmer Joe, the grass tuft and the wheat stalk come from farm_parts.npz/.json, exported
from the Blender scene once by tools/export_farm_parts.py.
"""
import atexit, ctypes, json, math, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import moderngl
from PIL import Image

W, H = 1080, 1920
PITCH = math.radians(54)
GAP = 0.05
FOC = 1500.0
BOARD_Y = 1000
HERE = Path(__file__).resolve().parent

def lin(rgb):
    c = np.array(rgb, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

THEMES = {   # the farm theme, as in Science_vid/blender/scene.py
    "farm": dict(floor=(104, 170, 62), floor_fresh=(196, 236, 90), line=(70, 128, 40), line_k=0.35,
                 block=(120, 80, 48), block_fresh=(84, 52, 30), block_h=0.14, world=(178, 210, 240), sun=2.6,
                 sun_col=(255, 244, 222), dot=(255, 214, 64),
                 crops=[((120, 80, 48), (84, 52, 30), 0.14), ((222, 178, 70), (255, 222, 120), 0.26),
                        ((66, 116, 44), (110, 170, 70), 0.14)]),
}
SUN_DIR = np.array([-0.45, 0.65, 1.0]); SUN_DIR /= np.linalg.norm(SUN_DIR)

def gentle():
    if sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)

# ---------- maths ----------
def euler(e):
    """Blender XYZ Euler -> 3x3 matrix."""
    cx, sx = math.cos(e[0]), math.sin(e[0]); cy, sy = math.cos(e[1]), math.sin(e[1]); cz, sz = math.cos(e[2]), math.sin(e[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def trs(loc, rot, scale):
    M = np.eye(4); M[:3, :3] = euler(rot) * np.asarray(scale, float)[None, :]; M[:3, 3] = loc
    return M

def look_at(eye, target, up=(0, 0, 1)):
    f = target - eye; f = f / np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    M = np.eye(4); M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M

def perspective(near=0.05, far=6000.0):
    P = np.zeros((4, 4))
    P[0, 0] = 2 * FOC / W; P[1, 1] = 2 * FOC / H
    P[1, 2] = 2 * (BOARD_Y - H / 2) / H            # Blender's lens shift: target lands on row BOARD_Y
    P[2, 2] = (far + near) / (near - far); P[2, 3] = 2 * far * near / (near - far); P[3, 2] = -1
    return P

def ortho(e, near=1.0, far=400.0):
    P = np.eye(4); P[0, 0] = P[1, 1] = 1 / e
    P[2, 2] = -2 / (far - near); P[2, 3] = -(far + near) / (far - near)
    return P

gl = lambda M: np.ascontiguousarray(M.T, "f4").tobytes()

def bevelled_box(hx=0.45, hy=0.45, hz=0.5, r=0.06, seg=3):
    """Science_vid's block: rounded edges, base at z=0, height 1 (instances scale the height)."""
    half = np.array([hx, hy, hz])
    def axis(h):
        inner = h - r; arc = [inner + r * k / seg for k in range(1, seg + 1)]
        return np.array([-a for a in arc[::-1]] + [-inner, inner] + arc)
    verts, idx = [], []
    for a in range(3):
        u, v = [k for k in range(3) if k != a]
        for s in (-1, 1):
            U, V = np.meshgrid(axis(half[u]), axis(half[v]), indexing="ij")
            p = np.zeros(U.shape + (3,)); p[..., a] = s * half[a]; p[..., u] = U; p[..., v] = V
            inner = np.clip(p, -(half - r), half - r)
            d = p - inner; n = d / np.linalg.norm(d, axis=-1, keepdims=True)
            q = inner + n * r; q[..., 2] += hz
            base = len(verts); nu, nv = U.shape
            verts += [np.concatenate([q[i, j], n[i, j]]) for i in range(nu) for j in range(nv)]
            for i in range(nu - 1):
                for j in range(nv - 1):
                    k = base + i * nv + j
                    idx += [k, k + nv, k + 1, k + 1, k + nv, k + nv + 1]
    return np.array(verts, "f4"), np.array(idx, "i4")

# ---------- shaders ----------
LIGHT = """
uniform vec3 u_L, u_sun, u_amb, u_eye;
uniform sampler2DShadow u_shadow; uniform float u_texel;
in vec4 v_ls; in vec3 v_world;
float sun_visible() {
    vec3 p = v_ls.xyz / v_ls.w * 0.5 + 0.5;
    if (p.x <= 0.0 || p.x >= 1.0 || p.y <= 0.0 || p.y >= 1.0 || p.z >= 1.0) return 1.0;
    float s = 0.0;
    for (int x = -1; x <= 1; x++) for (int y = -1; y <= 1; y++)
        s += texture(u_shadow, vec3(p.xy + vec2(x, y) * u_texel * 1.5, p.z - 0.0005));
    return s / 9.0;
}
vec3 light(vec3 col, vec3 n) {
    float sun = max(dot(n, u_L), 0.0) * sun_visible();
    vec3 v = normalize(u_eye - v_world);              // the sky, reflected at low angles (Fresnel)
    float fres = 0.04 + 0.96 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    return col * (u_sun * sun + u_amb * (0.65 + 0.35 * n.z)) + u_amb * fres * 0.3;
}
vec3 to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
"""
EULER = """
mat3 euler(vec3 e) {
    float cx = cos(e.x), sx = sin(e.x), cy = cos(e.y), sy = sin(e.y), cz = cos(e.z), sz = sin(e.z);
    mat3 Rx = mat3(1, 0, 0, 0, cx, sx, 0, -sx, cx);
    mat3 Ry = mat3(cy, 0, -sy, 0, 1, 0, sy, 0, cy);
    mat3 Rz = mat3(cz, sz, 0, -sz, cz, 0, 0, 0, 1);
    return Rz * Ry * Rx;
}
"""
CELLS_VS = """#version 330
uniform mat4 u_vp; uniform mat4 u_light; uniform float u_noff;
in vec3 in_pos; in vec3 in_nrm;
in vec3 i_pos; in vec3 i_scl; in float i_heat; in float i_kind;
out vec3 v_world; out vec3 v_obj; out vec3 v_nrm; out vec4 v_ls; out float v_heat; flat out float v_kind;
void main() {
    vec3 w = i_pos + in_pos * i_scl;
    v_world = w; v_obj = in_pos; v_heat = i_heat; v_kind = i_kind;
    v_nrm = normalize(in_nrm / i_scl);
    v_ls = u_light * vec4(w + v_nrm * u_noff, 1.0);
    gl_Position = u_vp * vec4(w, 1.0);
}
"""
CELLS_FS = "#version 330\n" + LIGHT + """
uniform float u_line_w, u_line_fade;
uniform vec3 c_floor, c_floor_fresh, c_line, c_ga, c_gb, c_grain, c_streak;
uniform vec3 c_crop[4]; uniform vec3 c_fresh[4];
in vec3 v_obj; in vec3 v_nrm; in float v_heat; flat in float v_kind;
out vec4 frag;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    vec3 n = normalize(v_nrm), col;
    int k = int(v_kind + 0.5);
    if (v_kind < -0.5) {                                        // grass floor with grid lines
        vec2 g = v_world.xy;
        float aa = max(fwidth(g.x), fwidth(g.y));
        float vary = mix(hash(floor(g)), 0.5, smoothstep(0.15, 0.6, aa));
        col = mix(c_ga, c_gb, vary * 0.6);
        vec2 d = abs(fract(g + 0.5) - 0.5);
        float line = 1.0 - smoothstep(u_line_w - aa, u_line_w + aa, min(d.x, d.y));
        col = mix(col, c_line, line * u_line_fade);
    } else if (k == 0) {
        col = mix(c_floor, c_floor_fresh, v_heat);             // freshly flipped grass tile
    } else {
        col = mix(c_crop[k], c_fresh[k], v_heat);
        if (k == 1) {                                          // dirt: grain and furrows
            col = mix(col, c_grain, vnoise(v_obj.xy * 18.0) * 0.5);
            float furrow = 0.5 + 0.5 * sin((v_obj.x * 2.2 + vnoise(v_obj.xy * 3.0) * 0.4) * 12.566);
            col *= mix(0.82, 1.0, furrow);
        } else if (k == 2) {                                   // wheat: fine golden streaks
            float st = pow(0.5 + 0.5 * sin((v_obj.y * 9.0 + vnoise(v_obj.xy * vec2(3.0, 12.0)) * 0.7) * 6.2832), 3.0);
            col = mix(col, c_streak, st * 0.35);
        }
    }
    frag = vec4(to_srgb(light(col, n)), 1.0);
}
"""
PROPS_VS = "#version 330\n" + EULER + """
uniform mat4 u_vp; uniform mat4 u_light; uniform float u_noff;
in vec3 in_pos; in vec3 in_nrm; in vec3 i_pos; in vec3 i_scl; in vec3 i_rot; in float i_rand;
out vec3 v_nrm; out vec4 v_ls; out float v_rand; out vec3 v_world;
void main() {
    mat3 R = euler(i_rot);
    vec3 w = i_pos + R * (in_pos * i_scl); v_world = w;
    v_nrm = normalize(R * (in_nrm / i_scl)); v_rand = i_rand;
    v_ls = u_light * vec4(w + v_nrm * u_noff, 1.0);
    gl_Position = u_vp * vec4(w, 1.0);
}
"""
PROPS_FS = "#version 330\n" + LIGHT + """
uniform vec3 u_ca, u_cb; uniform float u_emit;
in vec3 v_nrm; in float v_rand; out vec4 frag;
void main() {
    vec3 n = normalize(v_nrm); if (!gl_FrontFacing) n = -n;
    vec3 col = mix(u_ca, u_cb, v_rand);
    frag = vec4(to_srgb(light(col, n) + col * u_emit), 1.0);
}
"""
PARTS_VS = """#version 330
uniform mat4 u_vp; uniform mat4 u_light; uniform mat4 u_model; uniform mat3 u_nmat; uniform float u_noff;
in vec3 in_pos; in vec3 in_nrm;
out vec3 v_nrm; out vec4 v_ls; out vec3 v_world;
void main() {
    vec3 w = (u_model * vec4(in_pos, 1.0)).xyz; v_world = w;
    v_nrm = normalize(u_nmat * in_nrm);
    v_ls = u_light * vec4(w + v_nrm * u_noff, 1.0);
    gl_Position = u_vp * vec4(w, 1.0);
}
"""
PARTS_FS = "#version 330\n" + LIGHT + """
uniform vec3 u_col; uniform float u_emit;
in vec3 v_nrm; out vec4 frag;
void main() {
    vec3 n = normalize(v_nrm); if (!gl_FrontFacing) n = -n;
    frag = vec4(to_srgb(light(u_col, n) + u_col * u_emit), 1.0);
}
"""
SH_CELLS = """#version 330
uniform mat4 u_light; in vec3 in_pos; in vec3 i_pos; in vec3 i_scl;
void main() { gl_Position = u_light * vec4(i_pos + in_pos * i_scl, 1.0); }
"""
SH_PROPS = "#version 330\n" + EULER + """
uniform mat4 u_light; in vec3 in_pos; in vec3 i_pos; in vec3 i_scl; in vec3 i_rot;
void main() { gl_Position = u_light * vec4(i_pos + euler(i_rot) * (in_pos * i_scl), 1.0); }
"""
SH_PARTS = """#version 330
uniform mat4 u_light; uniform mat4 u_model; in vec3 in_pos;
void main() { gl_Position = u_light * (u_model * vec4(in_pos, 1.0)); }
"""
SH_FS = "#version 330\nvoid main() {}\n"

class _Vec(tuple):
    x = property(lambda s: s[0]); y = property(lambda s: s[1]); z = property(lambda s: s[2])

_CURRENT = None

def world_to_camera_view(scene, cam, co):
    """Same as Blender's: (x, y) from 0 to 1 across the frame (y up), z = distance."""
    return _CURRENT.to_camera_view(co)

class Scene:
    def __init__(self, theme="farm", scale=100, threads=4, engine="eevee", samples=0, character=None):
        global _CURRENT
        if theme != "farm" or character not in (None, "joe"):
            raise NotImplementedError("the fast renderer only knows the farm theme with Farmer Joe so far")
        _CURRENT = self
        self.theme = theme; self.T = T = THEMES[theme]
        self.heights = [0.0] + [c[2] for c in T["crops"]]
        self.sc = self.cam = None                        # Blender objects in the original; unused here
        self.w, self.h = W * scale // 100, H * scale // 100
        self.ctx = ctx = moderngl.create_standalone_context()
        self._programs(); self._geometry(); self._targets()
        self.cells_n = 0; self.tufts_n = self.stalks_n = 0
        self.line_w, self.line_fade = 0.025, T["line_k"]
        self.parts_now = []
        self.pool = ThreadPoolExecutor(max_workers=2); self.pending = []
        atexit.register(self._finish)

    # ---------- setup ----------
    def _programs(self):
        ctx, T = self.ctx, self.T
        self.p_cells = ctx.program(vertex_shader=CELLS_VS, fragment_shader=CELLS_FS)
        self.p_props = ctx.program(vertex_shader=PROPS_VS, fragment_shader=PROPS_FS)
        self.p_parts = ctx.program(vertex_shader=PARTS_VS, fragment_shader=PARTS_FS)
        self.s_cells = ctx.program(vertex_shader=SH_CELLS, fragment_shader=SH_FS)
        self.s_props = ctx.program(vertex_shader=SH_PROPS, fragment_shader=SH_FS)
        self.s_parts = ctx.program(vertex_shader=SH_PARTS, fragment_shader=SH_FS)
        for p in (self.p_cells, self.p_props, self.p_parts):
            p["u_L"].value = tuple(SUN_DIR)
            p["u_sun"].value = tuple(lin(T["sun_col"]) * T["sun"] / math.pi)
            p["u_amb"].value = tuple(lin(T["world"]) * 0.35)
            p["u_shadow"].value = 0
        pc = self.p_cells
        for k, v in dict(c_floor=T["floor"], c_floor_fresh=T["floor_fresh"], c_line=T["line"], c_ga=(92, 156, 54),
                         c_gb=(126, 186, 70), c_grain=(70, 44, 26), c_streak=(176, 128, 40)).items():
            pc[k].value = tuple(lin(v))
        crops = [T["floor"]] + [c[0] for c in T["crops"]]; fresh = [T["floor_fresh"]] + [c[1] for c in T["crops"]]
        pc["c_crop"].write(np.array([lin(c) for c in crops], "f4").tobytes())
        pc["c_fresh"].write(np.array([lin(c) for c in fresh], "f4").tobytes())

    def _geometry(self):
        ctx = self.ctx
        bv, bi = bevelled_box()
        box_vbo, box_ibo = ctx.buffer(bv.tobytes()), ctx.buffer(bi.tobytes())
        self.cells_vbo = ctx.buffer(reserve=400_000 * 32, dynamic=True)
        self.v_cells = ctx.vertex_array(self.p_cells, [(box_vbo, "3f 3f", "in_pos", "in_nrm"),
                       (self.cells_vbo, "3f 3f 1f 1f/i", "i_pos", "i_scl", "i_heat", "i_kind")], box_ibo)
        self.vs_cells = ctx.vertex_array(self.s_cells, [(box_vbo, "3f 3x4", "in_pos"),
                        (self.cells_vbo, "3f 3f 2x4/i", "i_pos", "i_scl")], box_ibo)
        S = 4000.0
        fv = np.array([[-S, -S, 0, 0, 0, 1], [S, -S, 0, 0, 0, 1], [S, S, 0, 0, 0, 1], [-S, S, 0, 0, 0, 1]], "f4")
        self.v_floor = ctx.vertex_array(self.p_cells, [(ctx.buffer(fv.tobytes()), "3f 3f", "in_pos", "in_nrm"),
                       (ctx.buffer(np.array([0, 0, 0, 1, 1, 1, 0, -1], "f4").tobytes()), "3f 3f 1f 1f/i",
                        "i_pos", "i_scl", "i_heat", "i_kind")], ctx.buffer(np.array([0, 1, 2, 0, 2, 3], "i4").tobytes()))
        meta = json.loads((HERE / "farm_parts.json").read_text())
        arr = np.load(HERE / "farm_parts.npz")
        def mesh(name):
            return ctx.buffer(np.concatenate([arr[name + ":pos"], arr[name + ":nrm"]], 1).astype("f4").tobytes())
        # grass tufts and wheat stalks: one small mesh each, instanced
        self.props = {}
        for name, cap in (("Tuft", 700_000), ("Stalk", 300_000)):
            vbo = ctx.buffer(reserve=cap * 40, dynamic=True); m = mesh(name)
            self.props[name] = dict(vbo=vbo,
                vao=ctx.vertex_array(self.p_props, [(m, "3f 3f", "in_pos", "in_nrm"),
                    (vbo, "3f 3f 3f 1f/i", "i_pos", "i_scl", "i_rot", "i_rand")]),
                svao=ctx.vertex_array(self.s_props, [(m, "3f 3x4", "in_pos"),
                    (vbo, "3f 3f 3f 1x4/i", "i_pos", "i_scl", "i_rot")]))
        # Farmer Joe: a tree of parts, posed every frame like the Blender objects
        self.nodes = {}
        for d in meta["nodes"]:
            d = dict(d, children=[])
            if d["mesh"]:
                m = mesh(d["name"])
                d["vao"] = ctx.vertex_array(self.p_parts, [(m, "3f 3f", "in_pos", "in_nrm")])
                d["svao"] = ctx.vertex_array(self.s_parts, [(m, "3f 3x4", "in_pos")])
            self.nodes[d["name"]] = d
            if d["parent"]: self.nodes[d["parent"]]["children"].append(d["name"])
        self.roles = meta["roles"]
        for d in self.nodes.values():
            d["pose"] = (list(d["loc"]), list(d["rot"]), list(d["scale"]))

    def _targets(self):
        ctx = self.ctx
        self.SM = 2048
        self.shadow_tex = ctx.depth_texture((self.SM, self.SM)); self.shadow_tex.compare_func = "<="
        self.shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)
        self.msaa_fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((self.w, self.h), 3, samples=4)],
                                        depth_attachment=ctx.depth_renderbuffer((self.w, self.h), samples=4))
        self.out_fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((self.w, self.h), 3)])

    # ---------- per frame (same calls as the Blender scene) ----------
    def set_cells(self, xs, ys, hz, heat, kind, paint=None):
        if paint is not None: raise NotImplementedError("painted trails aren't supported yet")
        n = len(xs); kind = np.asarray(kind, float)
        hz = np.where(kind > 0, np.maximum(hz, 0.008), 0.006)
        one = np.ones(n)
        inst = np.stack([np.asarray(xs) + 0.5, -(np.asarray(ys) + 0.5), 0 * one, one, one, hz, heat, kind], -1)
        self.cells_vbo.write(np.ascontiguousarray(inst, "f4").tobytes()); self.cells_n = n

    def _write_props(self, name, pos, scl, rot, rnd):
        inst = np.concatenate([pos, scl, rot, rnd[:, None]], 1)
        self.props[name]["vbo"].write(np.ascontiguousarray(inst, "f4").tobytes())
        return len(inst)

    def set_tufts(self, grid, cx, cy, s_px, per=8, hz=None):
        """Grass tufts on grass squares and wheat stalks on wheat squares near the camera,
        only while zoomed in enough to see them (the same hashes as the Blender scene)."""
        self.tufts_n = self.stalks_n = 0
        if s_px <= 30: return
        R = int(1.3 * max(W, H) / s_px) + 2
        gx, gy = np.meshgrid(np.arange(int(cx) - R, int(cx) + R), np.arange(int(cy) - R, int(cy) + R))
        gx, gy = gx.ravel(), gy.ravel()
        inside = (gx >= 0) & (gx < grid.shape[1]) & (gy >= 0) & (gy < grid.shape[0])
        gx, gy = gx[inside], gy[inside]
        wheat = grid[gy, gx] == 2
        if wheat.any():
            wx, wy = gx[wheat], gy[wheat]
            z = hz[wy, wx] if hz is not None else np.full(len(wx), self.heights[2])
            k = np.arange(7)
            h = lambda s: (np.sin((wx[:, None] * 12.9898 + wy[:, None] * 78.233 + k[None, :] * 37.719 + s) * 1.7) * 43758.5453) % 1.0
            tx = (wx[:, None] + 0.14 + 0.72 * h(0.4)).ravel(); ty = (wy[:, None] + 0.14 + 0.72 * h(1.9)).ravel()
            sc_ = (0.8 + 0.5 * h(2.7)).ravel()
            rot = np.stack([(h(4.6).ravel() - 0.5) * 0.4, (h(5.1).ravel() - 0.5) * 0.4, h(6.8).ravel() * 2 * math.pi], -1)
            self.stalks_n = self._write_props("Stalk", np.stack([tx, -ty, np.repeat(z, 7)], -1),
                                              np.stack([sc_] * 3, -1), rot, h(8.3).ravel())
        keep = grid[gy, gx] == 0
        gx, gy = gx[keep], gy[keep]
        k = np.arange(per)
        h = lambda s: (np.sin((gx[:, None] * 12.9898 + gy[:, None] * 78.233 + k[None, :] * 37.719 + s) * 1.7) * 43758.5453) % 1.0
        tx = (gx[:, None] + 0.08 + 0.84 * h(0.0)).ravel(); ty = (gy[:, None] + 0.08 + 0.84 * h(1.3)).ravel()
        s = (0.7 + 0.7 * h(2.1)).ravel(); sz = s * (0.8 + 0.5 * h(3.7)).ravel()
        rot = np.stack([(h(4.2).ravel() - 0.5) * 0.3, (h(5.9).ravel() - 0.5) * 0.3, h(6.4).ravel() * 2 * math.pi], -1)
        self.tufts_n = self._write_props("Tuft", np.stack([tx, -ty, np.zeros(tx.size)], -1),
                                         np.stack([s, s, sz], -1), rot, h(7.7).ravel())

    def set_floor(self, s_px):
        self.line_fade = self.T["line_k"] * min(1.0, max(0.0, (s_px - 9) / 18))
        self.line_w = 0.025 * max(1.0, 30 / s_px)

    def set_walker(self, x, y, z, ang, s_px, walk=0.0, fall=0.0, hat=None, arms=None, tool=True):
        """Pose Joe exactly as the Blender scene does, then work out each part's world matrix."""
        nd, ro = self.nodes, self.roles
        nd[ro["root"]]["pose"] = ((x, -y, z), (0, 0, ang), (1, 1, 1))
        far = s_px <= 16
        dot = nd[ro["dot"]]; dot["pose"] = (dot["loc"], dot["rot"], (7 / s_px,) * 3)
        sw = math.sin(walk * math.pi * 2) * (1 - walk) * 0.5
        for i, hip in enumerate(ro["legs"]):
            nd[hip]["pose"] = (nd[hip]["loc"], (0, sw * (1 if i else -1), 0), nd[hip]["scale"])
        body = nd[ro["body"]]
        b_loc = (0, 0, abs(math.sin(walk * math.pi)) * 0.04 * (1 - walk)); b_rot = (0, fall * math.radians(84), 0)
        body["pose"] = (b_loc, b_rot, body["scale"])
        hn = nd[ro["hat"]]
        hn["pose"] = (b_loc, b_rot, hn["scale"]) if hat is None else (hat[0], hat[1], hn["scale"])
        for k, (sh, _) in ro["arms"].items():
            swa, out = (arms or {}).get(k, (0.0, 0.0))
            nd[sh]["pose"] = (nd[sh]["loc"], ((1 if k == "left" else -1) * out, swa, 0), nd[sh]["scale"])
        hidden = set()
        def subtree(name):
            for c in nd[name]["children"]: hidden.add(c); subtree(c)
        if far: subtree(ro["body"]); subtree(ro["hat"])
        else: hidden.add(ro["dot"])
        if far or not tool: hidden.update(ro["tool"])
        world = {}; self.parts_now = []
        for name, d in nd.items():                     # parents come before children (export order)
            M = trs(*d["pose"])
            world[name] = world[d["parent"]] @ M if d["parent"] else M
            if d["mesh"] and name not in hidden:
                self.parts_now.append((d, world[name]))

    def set_camera(self, cx, cy, s_px, yaw_deg, pitch_deg=None, tz=0.0):
        yaw = math.radians(yaw_deg); pitch = PITCH if pitch_deg is None else math.radians(pitch_deg)
        fwd = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), -math.sin(pitch)])
        target = np.array([cx, -cy, tz])
        self.eye = target - fwd * (FOC / s_px)
        self.view = look_at(self.eye, target)
        self.vp = perspective() @ self.view
        e = min(400.0, 1.6 * H / s_px)                 # the sun's view: a box around what the camera sees
        centre = target + np.array([fwd[0], fwd[1], 0]) * e * 0.35
        self.light = ortho(e) @ look_at(centre + SUN_DIR * 200, centre)
        self.noff = 1.5 * 2 * e / self.SM

    def to_camera_view(self, co):
        c = self.vp @ np.array([co[0], co[1], co[2], 1.0])
        return _Vec(((c[0] / c[3] + 1) / 2, (c[1] / c[3] + 1) / 2, c[3]))

    # ---------- drawing ----------
    def _draw(self):
        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST)
        # 1. shadow map: everything that casts a shadow, seen from the sun
        self.shadow_fbo.use(); self.shadow_fbo.clear(depth=1.0)
        L = gl(self.light)
        for p in (self.s_cells, self.s_props, self.s_parts): p["u_light"].write(L)
        if self.cells_n: self.vs_cells.render(instances=self.cells_n)
        for name, n in (("Tuft", self.tufts_n), ("Stalk", self.stalks_n)):
            if n: self.props[name]["svao"].render(instances=n)
        for d, M in self.parts_now:
            self.s_parts["u_model"].write(gl(M)); d["svao"].render()
        # 2. the frame itself
        self.msaa_fbo.use(); self.msaa_fbo.clear(*(np.array(self.T["world"]) / 255), 1.0)
        self.shadow_tex.use(0)
        VP = gl(self.vp)
        for p in (self.p_cells, self.p_props, self.p_parts):
            p["u_vp"].write(VP); p["u_light"].write(L)
            p["u_noff"].value = self.noff; p["u_texel"].value = 1.0 / self.SM; p["u_eye"].value = tuple(self.eye)
        pc = self.p_cells
        pc["u_line_fade"].value = self.line_fade; pc["u_line_w"].value = self.line_w
        self.v_floor.render()
        if self.cells_n: self.v_cells.render(instances=self.cells_n)
        pp = self.p_props
        if self.tufts_n:
            pp["u_ca"].value = tuple(lin((110, 180, 60))); pp["u_cb"].value = tuple(lin((175, 225, 95)))
            pp["u_emit"].value = 0.35; self.props["Tuft"]["vao"].render(instances=self.tufts_n)
        if self.stalks_n:
            pp["u_ca"].value = tuple(lin((214, 164, 60))); pp["u_cb"].value = tuple(lin((246, 210, 110)))
            pp["u_emit"].value = 0.0; self.props["Stalk"]["vao"].render(instances=self.stalks_n)
        pa = self.p_parts
        for d, M in self.parts_now:
            pa["u_model"].write(gl(M))
            pa["u_nmat"].write(np.ascontiguousarray(np.linalg.inv(M[:3, :3]), "f4").tobytes())  # (M^-1)^T, sent transposed
            pa["u_col"].value = tuple(d["colour"]); pa["u_emit"].value = d["emit"]
            d["vao"].render()
        ctx.copy_framebuffer(self.out_fbo, self.msaa_fbo)

    def render(self, path, fmt="JPEG"):
        self._draw()
        raw = self.out_fbo.read(components=3, alignment=1)
        w, h = self.w, self.h
        def save():
            im = Image.frombytes("RGB", (w, h), raw).transpose(Image.FLIP_TOP_BOTTOM)
            if fmt == "JPEG": im.save(str(path), quality=94)
            else: im.save(str(path))
        self.pending.append(self.pool.submit(save))     # JPEG encoding runs while the next frame is drawn
        while len(self.pending) > 4: self.pending.pop(0).result()

    def _finish(self):
        for f in self.pending: f.result()
        self.pool.shutdown(wait=True)
