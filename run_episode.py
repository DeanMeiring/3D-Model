"""Make a whole Science_vid episode with our renderer instead of Blender.

  python run_episode.py ep02_wheat

1. Builds a sandbox copy of Science_vid in out/sandbox (scripts, episode files, fonts, and the
   episode's voice + timing), with blender/scene.py swapped for our fastscene/scene.py.
   Science_vid itself is only read, never written.
2. Runs Science_vid's own blender/render_video.py on our renderer (no Blender).
3. Runs Science_vid's own pipeline/compose.py for captions, rule cards and sound.
The finished video is copied to out/<episode>.mp4.
"""
import argparse, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("episode", help="e.g. ep02_wheat")
ap.add_argument("--science", default=str(HERE.parent / "Science_vid"))
ap.add_argument("--skip-render", action="store_true")
ap.add_argument("--test-every", type=int, default=0, help="only render every n-th frame (to frames_draft), no compose")
args = ap.parse_args()

SV = Path(args.science).resolve(); BOX = HERE / "out" / "sandbox"; ep = args.episode
src_ep, box_ep = SV / "renders" / ep, BOX / "renders" / ep
for need in (SV / "episodes" / f"{ep}.py", src_ep / "timing.json", src_ep / "voice.wav"):
    if not need.exists(): sys.exit(f"missing {need}: run Science_vid's pipeline/voice.py for this episode first")

# 1. the sandbox
for d in ("pipeline", "episodes"):
    shutil.copytree(SV / d, BOX / d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(SV / "assets" / "fonts", BOX / "assets" / "fonts", dirs_exist_ok=True)
(BOX / "blender").mkdir(parents=True, exist_ok=True)
shutil.copy2(SV / "blender" / "render_video.py", BOX / "blender")
for f in ("scene.py",): shutil.copy2(HERE / "fastscene" / f, BOX / "blender" / f)
for f in ("farm_parts.npz", "farm_parts.json"): shutil.copy2(HERE / "assets" / f, BOX / "blender" / f)
box_ep.mkdir(parents=True, exist_ok=True)
for f in ("timing.json", "voice.wav"): shutil.copy2(src_ep / f, box_ep / f)
if (src_ep / "voice").exists(): shutil.copytree(src_ep / "voice", box_ep / "voice", dirs_exist_ok=True)

times = {}
if args.test_every:
    subprocess.run([sys.executable, str(HERE / "fastscene" / "launch.py"), str(BOX / "blender" / "render_video.py"),
                    "--", str(BOX / "episodes" / f"{ep}.py"), "--pause", "0", "--every", str(args.test_every)], check=True)
    sys.exit(f"test frames in {box_ep / 'frames_draft'}")
# 2. frames, from Science_vid's own render script on our renderer
if not args.skip_render:
    shutil.rmtree(box_ep / "frames", ignore_errors=True)
    t = time.time()
    subprocess.run([sys.executable, str(HERE / "fastscene" / "launch.py"), str(BOX / "blender" / "render_video.py"),
                    "--", str(BOX / "episodes" / f"{ep}.py"), "--pause", "0"], check=True)
    times["render frames"] = time.time() - t

# 3. captions, cards and sound, from Science_vid's own compose script (and its Python)
py = SV / ".venv" / "Scripts" / "python.exe"
t = time.time()
subprocess.run([str(py if py.exists() else sys.executable), str(BOX / "pipeline" / "compose.py"),
                str(BOX / "episodes" / f"{ep}.py")], check=True)
times["compose (captions + sound)"] = time.time() - t

final = HERE / "out" / f"{ep}.mp4"
shutil.copy2(box_ep / f"{ep}.mp4", final)
print()
for k, v in times.items(): print(f"{k}: {v / 60:.1f} min ({v:.0f} s)")
print(f"video: {final}")
