"""Run Visual-fighting's blender/render_fight.py without Blender.

  python fastarena/launch.py <sandbox>/blender/render_fight.py -- renders/<id> [its own options]

The script's folder must hold our arena.py, vmath.py and arena.npz/.json (run_fight.py puts
them there). bpy, mathutils and bpy_extras are filled in with small stand-ins.
"""
import runpy, sys, types
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))
import vmath  # noqa: E402

bpy = types.ModuleType("bpy")
bpy.context = types.SimpleNamespace(view_layer=types.SimpleNamespace(update=lambda: None))
bpy.ops = types.SimpleNamespace(render=types.SimpleNamespace(render=None))     # arena.Arena fills this in
mathutils = types.ModuleType("mathutils")
mathutils.Vector, mathutils.Euler, mathutils.Quaternion = vmath.Vector, vmath.Euler, vmath.Quaternion
bpy_extras = types.ModuleType("bpy_extras"); object_utils = types.ModuleType("bpy_extras.object_utils")
def world_to_camera_view(scene, cam, co):
    return sys.modules["arena"].world_to_camera_view(scene, cam, co)
object_utils.world_to_camera_view = world_to_camera_view; bpy_extras.object_utils = object_utils
sys.modules.update({"bpy": bpy, "mathutils": mathutils, "bpy_extras": bpy_extras,
                    "bpy_extras.object_utils": object_utils})

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
