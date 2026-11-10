"""Run once in Blender (headless) to copy Farmer Joe, the grass tuft and the wheat stalk out of
Science_vid's scene into a plain data file our renderer can load.

  blender -b -P tools/export_farm_parts.py -- ../Science_vid assets/farm_parts.npz

For every object under Joe's root: its parent, its rest pose (location, Euler rotation, scale)
and, for meshes, the triangles (local coordinates, smooth normals) and the material colour.
"""
import json, sys
from pathlib import Path
import numpy as np
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
SV, OUT = Path(argv[0]).resolve(), Path(argv[1]).resolve()
sys.path.insert(0, str(SV / "blender")); sys.path.insert(0, str(SV / "pipeline"))
import scene as S

sc = S.Scene("farm", scale=10)

def material(o):
    """(base colour linear RGB, emission strength) from the object's Principled BSDF."""
    if not o.material_slots or not o.material_slots[0].material: return [1, 0, 1], 0.0
    nt = o.material_slots[0].material.node_tree
    b = nt.nodes.get("Principled BSDF")
    col = list(b.inputs["Base Color"].default_value)[:3]
    return col, float(b.inputs["Emission Strength"].default_value)

def triangles(o):
    me = o.data; me.calc_loop_triangles()
    pos, nrm = [], []
    for t in me.loop_triangles:
        for li in t.loops:
            pos.append(tuple(me.vertices[me.loops[li].vertex_index].co))
            nrm.append(tuple(me.corner_normals[li].vector))
    return np.array(pos, "f4"), np.array(nrm, "f4")

nodes, arrays = [], {}
todo = [sc.root]
while todo:
    o = todo.pop(0)
    todo += sorted(o.children, key=lambda c: c.name)
    if o.type not in ("EMPTY", "MESH"): continue          # the glow light is skipped
    d = dict(name=o.name, parent=o.parent.name if o.parent else None,
             loc=list(o.location), rot=list(o.rotation_euler), scale=list(o.scale), mesh=o.type == "MESH")
    if o.type == "MESH":
        d["colour"], d["emit"] = material(o)
        arrays[o.name + ":pos"], arrays[o.name + ":nrm"] = triangles(o)
    nodes.append(d)

for name in ("Tuft", "Stalk"):                              # instanced on grass / wheat squares
    o = bpy.data.objects[name]
    arrays[name + ":pos"], arrays[name + ":nrm"] = triangles(o)

roles = dict(root=sc.root.name, body=sc.body.name, hat=sc.hat.name, dot=sc.dot.name,
             legs=[h.name for h in sc.legs], tool=[t.name for t in sc.tool],
             arms={k: [sh.name, list(rel)] for k, (sh, rel) in sc.arms.items()})
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, **arrays)
OUT.with_suffix(".json").write_text(json.dumps(dict(nodes=nodes, roles=roles), indent=1))
print(f"exported {len(nodes)} nodes, {sum(n['mesh'] for n in nodes)} meshes -> {OUT}")
