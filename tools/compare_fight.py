"""Side by side: EEVEE (Visual-fighting's half-size draft) on top, ours below, for some frames."""
import sys
from pathlib import Path
from PIL import Image
ev, ours, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
idx = [int(x) for x in sys.argv[4].split(",")]
w, h = 540, 960
im = Image.new("RGB", (w * len(idx), h * 2))
for k, i in enumerate(idx):
    a = ev / f"f_{i:05d}.png"
    if a.exists(): im.paste(Image.open(a).convert("RGB").resize((w, h)), (k * w, 0))
    im.paste(Image.open(ours / f"f_{i:05d}.png").convert("RGB").resize((w, h), Image.LANCZOS), (k * w, h))
im.save(out, quality=90)
