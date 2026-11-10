"""Run a Science_vid Blender script (e.g. blender/render_video.py) without Blender.

  python fastscene/launch.py <script.py> -- <the script's own arguments>

The script's folder must hold our scene.py (the sandbox made by run_episode.py does this).
The few Blender modules the script touches outside the scene (bpy.context.view_layer.update,
bpy_extras.object_utils.world_to_camera_view, mathutils.Vector) are filled in with stand-ins.
"""
import runpy, sys, types

class Vector(tuple):
    x = property(lambda s: s[0]); y = property(lambda s: s[1]); z = property(lambda s: s[2])

bpy = types.ModuleType("bpy")
bpy.context = types.SimpleNamespace(view_layer=types.SimpleNamespace(update=lambda: None))
bpy_extras = types.ModuleType("bpy_extras"); object_utils = types.ModuleType("bpy_extras.object_utils")
def world_to_camera_view(scene, cam, co):
    return sys.modules["scene"].world_to_camera_view(scene, cam, co)
object_utils.world_to_camera_view = world_to_camera_view; bpy_extras.object_utils = object_utils
mathutils = types.ModuleType("mathutils"); mathutils.Vector = Vector
sys.modules.update({"bpy": bpy, "bpy_extras": bpy_extras, "bpy_extras.object_utils": object_utils, "mathutils": mathutils})

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
