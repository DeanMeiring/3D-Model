# 3D-Model

Learning how graphics are rendered by building a small renderer, starting with a fast
renderer for Science_vid's block scenes.

## Prototype 1: a fast block renderer for Science_vid

**Question:** can a small renderer of our own (Python + OpenGL through moderngl, on the
laptop's Intel graphics) make Science_vid's frames much faster than Blender's EEVEE?

**Baseline (measured, from `Science_vid/renders/ep01_farm/render.log`):** EEVEE rendered
episode 1's 1,359 frames at 1080×1920 in 31.0 min, **1.37 s per frame**.

**Bar (set 10 November 2026, before the first run):**
- **Pass:** on average under **0.3 s per frame** at 1080×1920 over all of episode 1,
  including handing each frame to ffmpeg to make the video.
- **Fail:** 0.3 s per frame or more. Then the idea isn't worth pursuing.
- Reported but not part of the bar: how close the look is to EEVEE, judged on
  side-by-side frames.

**Scope of the prototype:** the same episode timeline, simulation and camera moves as
`blender/render_video.py`, with bevelled blocks, a grass floor with grid lines, and sun
shadows. Not included: grass tufts, the farmer model (a placeholder stands in for him),
captions and sound (those stay in `pipeline/compose.py`).

**Result (10 November 2026): PASS.** All 1,359 frames of episode 1, rendered and encoded
to `out/ep01_farm.mp4` (1080×1920, 30 fps, 45.3 s), in **24.4 s: 0.018 s per frame,
about 76× faster than EEVEE's 1.37 s**, at below-normal priority and with no rests.

| Per frame | Time |
|---|---|
| Update the scene (numpy) | 0.6 ms |
| Draw it on the GPU (shadow pass + main pass, 4× MSAA) | 4.0 ms |
| Copy the frame back from the GPU | 11.5 ms |
| Hand it to ffmpeg | 1.2 ms |

Copying the finished frame off the GPU now takes longer than drawing it.

**Look:** the camera, simulation, blocks and shadows match EEVEE frame for frame
(`out/compare_f*.jpg`, EEVEE left, ours right). Still missing: grass tufts, Farmer Joe
(a placeholder of boxes), the furrow bump on the dirt, and EEVEE's softer shadows.
Adding those will cost some speed, but there is about 16× headroom to the bar.

## Episode 2, made end to end (10 November 2026)

`fastscene/scene.py` is a drop-in replacement for Science_vid's `blender/scene.py` (same
calls), so Science_vid's own `blender/render_video.py` and `pipeline/compose.py` run
unchanged on it, in a sandbox copy (`out/sandbox`). Farmer Joe, the grass tuft and the
wheat stalk are exported from the Blender scene once (`tools/export_farm_parts.py`).

```
.venv/Scripts/python run_episode.py ep02_wheat     # -> out/ep02_wheat.mp4
```

| Step | EEVEE | Ours |
|---|---|---|
| Frames (1,883 at 1080×1920, with JPEGs) | ~45 min (1.4 s/frame), ~75 min with the 1 s rests | **63 s** (0.033 s/frame) |
| Captions, cards, sound (`compose.py`, unchanged) | ~3 min | 3.1 min |
| **Whole episode** | **~50–80 min** | **~4 min** |

Included now: three crops (wheat with streaks and stalks), grass tufts, the real Farmer Joe
with his walk, the trip (fall, hat flying off and back on), the camera swing, the comparison
with episode 1's field and its floating labels. Still missing: Joe's glow light when far
away, soft shadow edges as wide as EEVEE's, and the bump detail on dirt and wheat.
Only the farm theme with Joe is supported; other episodes (sheep, barns, painted
trails…) use their own render scripts and props.

## Visual-fighting's ball arena (10 November 2026)

`fastarena/arena.py` stands in for Visual-fighting's `blender/arena.py`: same objects, pools and
helpers, so its own `blender/render_fight.py` and `fight/compose.py` run unchanged (sandbox in
`out/vf_sandbox`). The arena's meshes, materials and lights are exported from Blender once
(`tools/export_arena.py`); the cast's colours and pictures are set per fight.

```
.venv/Scripts/python run_fight.py fight01                 # -> out/fight01.mp4
.venv/Scripts/python run_fight.py fight01 --frames 10,300 # just some frames, to check the look
```

| fight01 (825 frames, 27.5 s) | EEVEE | Ours |
|---|---|---|
| Frames | half size: 3.1 s/frame (`render_50.log`), ~43 min; full size not measured, several times that | **full size: 80 s** (0.1 s/frame) |
| Overlay + sound (`compose.py`, unchanged) | | 2.3 min |

Look: matched on side-by-side frames (`tools/compare_fight.py`). Included: area/spot lights,
soft key-light shadows, the floor's mirror image, ring pattern, emoji balls, see-through
effects, depth of field, bloom, lens fringe, AgX Punchy. Differences: EEVEE has a soft white
patch on the floor near the middle that ours lacks, and its glow is a little softer.

## Run it

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python block_renderer.py                 # all of episode 1 -> out/ep01_farm.mp4
.venv/Scripts/python block_renderer.py --stills-only   # only the side-by-side comparison frames
```

It reads the episode's timing and simulation from `../Science_vid` (read-only).
