"""Make a Visual-fighting comment-fight video with our renderer instead of Blender.

  python run_fight.py fight01                    # full size -> out/fight01.mp4
  python run_fight.py fight01 --frames 14,30,300 # just these frames (to check the look)

1. Builds a sandbox copy of Visual-fighting in out/vf_sandbox (fight/, assets/, the render script,
   the recorded fight), with blender/arena.py swapped for our fastarena/arena.py.
   Visual-fighting itself is only read, never written.
2. Runs Visual-fighting's own blender/render_fight.py on our renderer (no Blender).
3. Runs Visual-fighting's own fight/compose.py for the overlay and sound.
"""
import argparse, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("fight", help="a recorded fight in Visual-fighting/renders, e.g. fight01")
ap.add_argument("--vf", default=str(HERE.parent / "Visual-fighting"))
ap.add_argument("--scale", type=int, default=100)
ap.add_argument("--frames", default="", help="only render these frames, no compose")
ap.add_argument("--skip-render", action="store_true")
args = ap.parse_args()

VF = Path(args.vf).resolve(); BOX = HERE / "out" / "vf_sandbox"; fid = args.fight
src, dst = VF / "renders" / fid, BOX / "renders" / fid
if not (src / "fight.json").exists():
    sys.exit(f"missing {src / 'fight.json'}: record it first (python -m fight.record --out renders/{fid})")

for d in ("fight", "assets"):
    shutil.copytree(VF / d, BOX / d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
(BOX / "blender").mkdir(parents=True, exist_ok=True)
shutil.copy2(VF / "blender" / "render_fight.py", BOX / "blender")
for f in ("arena.py", "vmath.py"): shutil.copy2(HERE / "fastarena" / f, BOX / "blender" / f)
for f in ("arena.npz", "arena.json"): shutil.copy2(HERE / "assets" / f, BOX / "blender" / f)
dst.mkdir(parents=True, exist_ok=True)
shutil.copy2(src / "fight.json", dst)
shutil.copytree(src / "pics", dst / "pics", dirs_exist_ok=True)

frames = "frames" if args.scale == 100 else f"frames_{args.scale}"
render = [sys.executable, str(HERE / "fastarena" / "launch.py"), str(BOX / "blender" / "render_fight.py"), "--",
          str(dst), "--scale", str(args.scale)]
if args.frames:
    shutil.rmtree(dst / "stills", ignore_errors=True)
    for i in args.frames.split(","):
        (dst / frames / f"f_{int(i):05d}.png").unlink(missing_ok=True)
    subprocess.run(render + ["--frames", args.frames], check=True)
    sys.exit(f"frames in {dst / frames}")

times = {}
if not args.skip_render:
    shutil.rmtree(dst / frames, ignore_errors=True)
    t = time.time()
    subprocess.run(render, check=True)
    times["render frames"] = time.time() - t
t = time.time()
subprocess.run([sys.executable, "-m", "fight.compose", str(dst), "--frames", frames], check=True, cwd=BOX)
times["compose (overlay + sound)"] = time.time() - t
out = sorted(dst.glob("*.mp4"), key=lambda p: p.stat().st_mtime)[-1]
final = HERE / "out" / f"{fid}.mp4"
shutil.copy2(out, final)
print()
for k, v in times.items(): print(f"{k}: {v / 60:.1f} min ({v:.0f} s)")
print(f"video: {final}")
