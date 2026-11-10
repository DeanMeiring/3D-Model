"""Prototype 1: a fast renderer for Science_vid's farm scenes (Python + OpenGL via moderngl).

Replays an episode's real timeline (timing.json), simulation and camera moves from
Science_vid/blender/render_video.py, but draws each frame in one GPU pass:
  1. shadow pass: the blocks seen from the sun, into a depth texture (a shadow map)
  2. main pass: floor + every block as instances of one bevelled-box mesh, 4x MSAA
  3. the finished frame is read back and piped straight into ffmpeg (no JPEGs on disk)

Usage:
  python block_renderer.py                      all of episode 1 -> out/ep01_farm.mp4
  python block_renderer.py --stills-only        just the comparison frames vs EEVEE
"""
import argparse, ctypes, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import moderngl
from PIL import Image

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--science", default=str(HERE.parent / "Science_vid"), help="the Science_vid checkout (read only)")
ap.add_argument("--episode", default="ep01_farm")
ap.add_argument("--stills", default="0,200,500,800,1100,1350", help="frames saved next to the EEVEE frame")
ap.add_argument("--stills-only", action="store_true", help="skip the video, only save the stills")
ap.add_argument("--msaa", type=int, default=4)
args = ap.parse_args()

SV = Path(args.science)
sys.path.insert(0, str(SV / "pipeline"))
from common import FPS, N, load_episode, load_timeline, simulate   # Science_vid's own sim + timeline

W, H = 1080, 1920
PITCH = math.radians(54)
FOC = 1500.0        # focal length in pixels at 1080 wide (as Science_vid)
BOARD_Y = 1000      # pixel row the camera target lands on
GAP = 0.05
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)

# ---------- the farm theme (copied from Science_vid/blender/scene.py) ----------
def lin(rgb):
    c = np.array(rgb, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

T = dict(floor=(104, 170, 62), floor_fresh=(196, 236, 90), line=(70, 128, 40), line_k=0.35,
         block=(120, 80, 48), block_fresh=(84, 52, 30), world=(178, 210, 240), sun=2.6,
         sun_col=(255, 244, 222), dot=(255, 214, 64), grass_a=(92, 156, 54), grass_b=(126, 186, 70),
         grain=(70, 44, 26), body=(52, 82, 160), skin=(232, 186, 150), hat=(226, 196, 110))
HEIGHTS = [0.0, 0.14, 0.26, 0.14]
SUN_DIR = np.array([-0.45, 0.65, 1.0]); SUN_DIR /= np.linalg.norm(SUN_DIR)   # towards the sun

def gentle():
    """Below-normal priority (ffmpeg inherits it), as Science_vid does."""
    if sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)

# ---------- geometry ----------
def bevelled_box(hx=0.45, hy=0.45, hz=0.5, r=0.06, seg=3):
    """A box with rounded edges, base at z=0 and height 1 (instances scale it to the block
    height, like Science_vid's geometry-nodes block). Returns (vertices [pos, normal], indices)."""
    half = np.array([hx, hy, hz])
    def axis(h):
        inner = h - r
        arc = [inner + r * k / seg for k in range(1, seg + 1)]
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

# ---------- matrices (row-major numpy; sent transposed to GL) ----------
def look_at(eye, target, up=(0, 0, 1)):
    f = target - eye; f = f / np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    M = np.eye(4); M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M

def perspective(near=0.05, far=6000.0):
    P = np.zeros((4, 4))
    P[0, 0] = 2 * FOC / W; P[1, 1] = 2 * FOC / W * W / H
    P[1, 2] = 2 * (BOARD_Y - H / 2) / H          # Blender's shift_y: target sits at row BOARD_Y
    P[2, 2] = (far + near) / (near - far); P[2, 3] = 2 * far * near / (near - far); P[3, 2] = -1
    return P

def ortho(e, near=1.0, far=400.0):
    P = np.eye(4); P[0, 0] = P[1, 1] = 1 / e
    P[2, 2] = -2 / (far - near); P[2, 3] = -(far + near) / (far - near)
    return P

gl = lambda M: np.ascontiguousarray(M.T, "f4").tobytes()

# ---------- shaders ----------
VS = """
#version 330
uniform mat4 u_vp; uniform mat4 u_light; uniform float u_noff;
in vec3 in_pos; in vec3 in_nrm;
in vec3 i_pos; in vec3 i_scl; in float i_heat; in float i_kind;
out vec3 v_world; out vec3 v_obj; out vec3 v_nrm; out vec4 v_ls; out float v_heat; flat out float v_kind;
void main() {
    vec3 w = i_pos + in_pos * i_scl;
    v_world = w; v_obj = in_pos; v_heat = i_heat; v_kind = i_kind;
    v_nrm = normalize(in_nrm / i_scl);
    v_ls = u_light * vec4(w + v_nrm * u_noff, 1.0);    // normal offset: no shadow acne
    gl_Position = u_vp * vec4(w, 1.0);
}
"""
FS = """
#version 330
uniform vec3 u_L, u_sun, u_amb;
uniform sampler2DShadow u_shadow; uniform float u_texel;
uniform float u_line_w, u_line_fade;
uniform vec3 c_floor, c_floor_fresh, c_line, c_ga, c_gb, c_block, c_block_fresh, c_grain, c_body, c_skin, c_hat, c_dot;
in vec3 v_world; in vec3 v_obj; in vec3 v_nrm; in vec4 v_ls; in float v_heat; flat in float v_kind;
out vec4 frag;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
float sun_visible() {
    vec3 p = v_ls.xyz / v_ls.w * 0.5 + 0.5;
    if (p.x <= 0.0 || p.x >= 1.0 || p.y <= 0.0 || p.y >= 1.0 || p.z >= 1.0) return 1.0;
    float s = 0.0;                                   // 3x3 PCF: soft-ish shadow edges
    for (int x = -1; x <= 1; x++) for (int y = -1; y <= 1; y++)
        s += texture(u_shadow, vec3(p.xy + vec2(x, y) * u_texel * 1.5, p.z - 0.0005));
    return s / 9.0;
}
vec3 to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
void main() {
    vec3 n = normalize(v_nrm);
    vec3 col;
    if (v_kind > 12.5) { frag = vec4(to_srgb(c_dot * 1.4), 1.0); return; }      // far-away walker dot
    if (v_kind < -0.5) {                                                         // grass floor + grid lines
        vec2 g = v_world.xy;
        float aa = max(fwidth(g.x), fwidth(g.y));                                // cells per pixel
        // per-square shade; fades to the average when squares get too small to see (no shimmer)
        float vary = mix(hash(floor(g)), 0.5, smoothstep(0.15, 0.6, aa));
        col = mix(c_ga, c_gb, vary * 0.6);
        vec2 d = abs(fract(g + 0.5) - 0.5);                                      // distance to a grid line
        float line = 1.0 - smoothstep(u_line_w - aa, u_line_w + aa, min(d.x, d.y));
        col = mix(col, c_line, line * u_line_fade);
    } else if (v_kind < 0.5) {
        col = mix(c_floor, c_floor_fresh, v_heat);                               // freshly flipped grass tile
    } else if (v_kind < 1.5) {                                                   // dirt: furrows + grain
        col = mix(c_block, c_block_fresh, v_heat);
        col = mix(col, c_grain, vnoise(v_obj.xy * 18.0) * 0.5);
        float furrow = 0.5 + 0.5 * sin((v_obj.x * 2.2 + vnoise(v_obj.xy * 3.0) * 0.4) * 6.2832 * 2.0);
        col *= mix(0.82, 1.0, furrow);
    } else if (v_kind < 10.5) col = c_body;
    else if (v_kind < 11.5) col = c_skin;
    else col = c_hat;
    float sun = max(dot(n, u_L), 0.0) * sun_visible();
    vec3 lit = col * (u_sun * sun + u_amb * (0.65 + 0.35 * n.z));
    frag = vec4(to_srgb(lit), 1.0);
}
"""
SHADOW_VS = """
#version 330
uniform mat4 u_light;
in vec3 in_pos; in vec3 i_pos; in vec3 i_scl;
void main() { gl_Position = u_light * vec4(i_pos + in_pos * i_scl, 1.0); }
"""
SHADOW_FS = "#version 330\nvoid main() {}\n"

# ---------- GPU setup ----------
ctx = moderngl.create_standalone_context()
prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
sprog = ctx.program(vertex_shader=SHADOW_VS, fragment_shader=SHADOW_FS)
for k, v in dict(c_floor="floor", c_floor_fresh="floor_fresh", c_line="line", c_ga="grass_a", c_gb="grass_b",
                 c_block="block", c_block_fresh="block_fresh", c_grain="grain", c_body="body", c_skin="skin",
                 c_hat="hat", c_dot="dot").items():
    prog[k].value = tuple(lin(T[v]))
prog["u_L"].value = tuple(SUN_DIR)
# Blender's sun (strength 2.6, diffuse /pi) and world light (strength 0.35), roughly
prog["u_sun"].value = tuple(lin(T["sun_col"]) * T["sun"] / math.pi)
prog["u_amb"].value = tuple(lin(T["world"]) * 0.35)

bv, bi = bevelled_box()
box_vbo, box_ibo = ctx.buffer(bv.tobytes()), ctx.buffer(bi.tobytes())
MAX_INST = 400_000
inst_vbo = ctx.buffer(reserve=MAX_INST * 8 * 4, dynamic=True)
blocks = ctx.vertex_array(prog, [(box_vbo, "3f 3f", "in_pos", "in_nrm"),
                                 (inst_vbo, "3f 3f 1f 1f/i", "i_pos", "i_scl", "i_heat", "i_kind")], box_ibo)
blocks_shadow = ctx.vertex_array(sprog, [(box_vbo, "3f 3x4", "in_pos"),
                                         (inst_vbo, "3f 3f 2x4/i", "i_pos", "i_scl")], box_ibo)
S = 4000.0
floor_v = np.array([[-S, -S, 0, 0, 0, 1], [S, -S, 0, 0, 0, 1], [S, S, 0, 0, 0, 1], [-S, S, 0, 0, 0, 1]], "f4")
floor_inst = ctx.buffer(np.array([0, 0, 0, 1, 1, 1, 0, -1], "f4").tobytes())
floor = ctx.vertex_array(prog, [(ctx.buffer(floor_v.tobytes()), "3f 3f", "in_pos", "in_nrm"),
                                (floor_inst, "3f 3f 1f 1f/i", "i_pos", "i_scl", "i_heat", "i_kind")],
                         ctx.buffer(np.array([0, 1, 2, 0, 2, 3], "i4").tobytes()))

SM = 2048
shadow_tex = ctx.depth_texture((SM, SM)); shadow_tex.compare_func = "<="
shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR); shadow_tex.repeat_x = shadow_tex.repeat_y = False
shadow_fbo = ctx.framebuffer(depth_attachment=shadow_tex)
msaa_fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 3, samples=args.msaa)],
                           depth_attachment=ctx.depth_renderbuffer((W, H), samples=args.msaa))
out_fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 3)])
SKY = tuple(np.array(T["world"]) / 255)   # Science_vid's background: the world colour as shown

def draw(inst, cx, cy, s_px, yaw_deg):
    n = len(inst)
    inst_vbo.write(np.ascontiguousarray(inst, "f4").tobytes())
    yaw = math.radians(yaw_deg)
    fwd = np.array([math.sin(yaw) * math.cos(PITCH), math.cos(yaw) * math.cos(PITCH), -math.sin(PITCH)])
    target = np.array([cx, -cy, 0.0])
    view = look_at(target - fwd * (FOC / s_px), target)
    # sun's view: an orthographic box around what the camera sees
    e = min(400.0, 1.6 * H / s_px)
    centre = target + np.array([fwd[0], fwd[1], 0]) * e * 0.35
    light = ortho(e) @ look_at(centre + SUN_DIR * 200, centre)
    ctx.enable(moderngl.DEPTH_TEST)
    shadow_fbo.use(); shadow_fbo.clear(depth=1.0)
    sprog["u_light"].write(gl(light))
    blocks_shadow.render(instances=n)
    msaa_fbo.use(); msaa_fbo.clear(*SKY, 1.0)
    prog["u_vp"].write(gl(perspective() @ view)); prog["u_light"].write(gl(light))
    prog["u_noff"].value = 1.5 * 2 * e / SM; prog["u_texel"].value = 1.0 / SM
    prog["u_line_fade"].value = T["line_k"] * min(1.0, max(0.0, (s_px - 9) / 18))
    prog["u_line_w"].value = 0.025 * max(1.0, 30 / s_px)
    shadow_tex.use(0); prog["u_shadow"].value = 0
    floor.render()
    blocks.render(instances=n)
    ctx.copy_framebuffer(out_fbo, msaa_fbo)

# ---------- the episode: same timeline, sim and camera as render_video.py ----------
ep_dir = SV / "renders" / args.episode
ep = load_episode(SV / "episodes" / f"{args.episode}.py")
lines, DUR, NF, sched = load_timeline(ep_dir)
RULE = getattr(ep, "RULE", "RL"); K = len(RULE)
TOTAL = int(sched.max())
moves, hist = simulate(TOTAL, RULE)
mx = np.minimum.accumulate(moves[:, 0]); Mx = np.maximum.accumulate(moves[:, 0])
my = np.minimum.accumulate(moves[:, 1]); My = np.maximum.accumulate(moves[:, 1])
HGT = np.array(HEIGHTS[:K]); ANIM = 7; BOX = 900
ANGS = [math.pi / 2, 0, -math.pi / 2, math.pi]
stills = {int(s) for s in args.stills.split(",") if s.strip()}
last_frame = max(stills) + 1 if args.stills_only else NF

def walker(x, y, z, s_px):
    """Placeholder for Farmer Joe: a dot when far away (as Science_vid), a few boxes up close."""
    if s_px <= 16:
        d = 9 / s_px
        return [[x, -y, z + 0.1, d, d, d, 0, 13]]
    return [[x, -y, z, 0.32, 0.24, 0.55, 0, 10],          # body
            [x, -y, z + 0.55, 0.24, 0.24, 0.22, 0, 11],   # head
            [x, -y, z + 0.74, 0.62, 0.62, 0.035, 0, 12],  # hat brim
            [x, -y, z + 0.76, 0.3, 0.3, 0.12, 0, 12]]     # hat crown

gentle()
ff = None
video = OUT / f"{args.episode}.mp4"
if not args.stills_only:
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-vf", "vflip",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                           str(video)], stdin=subprocess.PIPE)

grid = np.zeros((N, N), np.uint8); prev = np.zeros((N, N), np.uint8)
last = np.full((N, N), -10 ** 9, np.int64); flip_f = np.full((N, N), -10 ** 9, np.int64)
cx = cy = N / 2 + 0.5; s_cur = 120.0
done = 0; step_f = 0
prev_pos = np.array(hist[0][:2], float); ang_prev = ANGS[0]
tm = dict(scene=0.0, gpu=0.0, read=0.0, ffmpeg=0.0); n_inst_max = 0
t_start = time.perf_counter()
for f in range(last_frame):
    t0 = time.perf_counter()
    tt = f / FPS; tgt = sched[f]
    if tgt > done:
        prev_pos = np.array(hist[done][:2], float) if tgt - done == 1 else np.array(hist[tgt][:2], float)
        ang_prev = ANGS[hist[done][2]] if tgt - done == 1 else ANGS[hist[tgt][2]]
        step_f = f
    while done < tgt:
        px_, py_ = moves[done]; prev[py_, px_] = grid[py_, px_]; grid[py_, px_] = (grid[py_, px_] + 1) % K
        last[py_, px_] = done; flip_f[py_, px_] = f; done += 1
    ax, ay, adir, under = hist[done]
    u = min(1.0, (f - step_f) / 6.0); u = 1 - (1 - u) ** 2
    pos = prev_pos * (1 - u) + np.array([ax, ay], float) * u
    if done == 0:
        bx0 = bx1 = ax; by0 = by1 = ay
    else:
        bx0, bx1 = min(mx[done - 1], ax), max(Mx[done - 1], ax); by0, by1 = min(my[done - 1], ay), max(My[done - 1], ay)
    span = max(bx1 - bx0, by1 - by0) + 1 + (3 if done < 30 else 8)
    s_t = min(140.0, BOX / span); k = 0.06 if done > 30 else 0.12
    tx_, ty_ = (bx0 + bx1 + 1) / 2, (by0 + by1 + 1) / 2
    s_cur *= (s_t / s_cur) ** k
    cx += (tx_ - cx) * k; cy += (ty_ - cy) * k
    if args.stills_only and f not in stills: continue
    rate = (sched[min(f + 2, NF - 1)] - sched[max(f - 2, 0)]) / 4 * FPS
    heat_len = max(1.2, rate * 0.06)
    sub = (slice(by0, by1 + 1), slice(bx0, bx1 + 1))
    cur, old = grid[sub], prev[sub]
    a = np.minimum(1.0, (f - flip_f[sub]) / ANIM); a = 1 - (1 - a) ** 2
    hz = HGT[old] * (1 - a) + HGT[cur] * a
    kind = np.where(cur > 0, cur, np.where(a < 1, old, 0))
    heat = np.exp(-(done - last[sub]) / heat_len) * 0.8
    ys, xs = np.nonzero((hz > 0.004) | (heat > 0.05) | (cur > 0))
    hz, heat, kind = hz[ys, xs], heat[ys, xs], kind[ys, xs]
    xs, ys = xs + bx0, ys + by0
    hz = np.where(kind > 0, np.maximum(hz, 0.008), 0.006)
    one = np.ones(len(xs))
    inst = np.stack([xs + 0.5, -(ys + 0.5), 0 * one, one, one, hz, heat, kind.astype(float)], -1)
    az = HGT[grid[ay, ax]] if (f - flip_f[ay, ax]) > ANIM else HGT[prev[ay, ax]]
    inst = np.concatenate([inst, np.array(walker(pos[0] + 0.5, pos[1] + 0.5, az, s_cur))])
    n_inst_max = max(n_inst_max, len(inst))
    t1 = time.perf_counter()
    draw(inst, cx, cy, s_cur, -18 + 36 * (tt / DUR))
    ctx.finish()
    t2 = time.perf_counter()
    raw = out_fbo.read(components=3, alignment=1)
    t3 = time.perf_counter()
    if ff: ff.stdin.write(raw)
    t4 = time.perf_counter()
    tm["scene"] += t1 - t0; tm["gpu"] += t2 - t1; tm["read"] += t3 - t2; tm["ffmpeg"] += t4 - t3
    if f in stills:
        ours = Image.frombytes("RGB", (W, H), raw).transpose(Image.FLIP_TOP_BOTTOM)
        ours.save(OUT / f"ours_f{f:05d}.png")
        ev = SV / "renders" / args.episode / "frames" / f"f_{f:05d}.jpg"
        if ev.exists():
            pair = Image.new("RGB", (W, H // 2 + 60), (20, 24, 32))
            pair.paste(Image.open(ev).resize((W // 2, H // 2)), (0, 60))
            pair.paste(ours.resize((W // 2, H // 2)), (W // 2, 60))
            from PIL import ImageDraw
            dr = ImageDraw.Draw(pair)
            dr.text((20, 20), f"EEVEE (Blender)   frame {f}", fill=(230, 230, 230))
            dr.text((W // 2 + 20, 20), f"Our renderer   frame {f}", fill=(230, 230, 230))
            pair.save(OUT / f"compare_f{f:05d}.jpg", quality=92)
    if f and f % 200 == 0 and not args.stills_only:
        el = time.perf_counter() - t_start
        print(f"frame {f}/{NF}  {el / (f + 1):.3f} s/frame", flush=True)

if ff:
    t5 = time.perf_counter(); ff.stdin.close(); ff.wait(); tm["ffmpeg"] += time.perf_counter() - t5
total = time.perf_counter() - t_start
if not args.stills_only:
    per = total / last_frame
    print(f"\n{last_frame} frames in {total:.1f} s = {per:.3f} s/frame ({1 / per:.1f} frames per second)")
    print("per frame: " + ", ".join(f"{k} {v / last_frame * 1000:.1f} ms" for k, v in tm.items()))
    print(f"most instances in one frame: {n_inst_max}  ({len(bi) // 3} triangles per block)")
    print(f"EEVEE baseline: 1.37 s/frame (31.0 min)  ->  {1.37 / per:.0f}x faster")
    print(f"bar (< 0.3 s/frame): {'PASS' if per < 0.3 else 'FAIL'}")
    print(f"video: {video}")
else:
    print(f"stills saved in {OUT}")
