"""Run once in Blender (headless) to copy Visual-fighting's 3D arena into a plain data file.

  blender -b -P tools/export_arena.py -- ../Visual-fighting/renders/fight01 assets/arena.npz

Builds the real arena (Visual-fighting/blender/arena.py) for that fight's cast, then saves every
mesh (triangles, corner normals, material per triangle), every object (mesh, materials, pose,
shown or hidden), every material's settings, the lights, and which objects make up which pool
(balls, held weapons, pickups, effects...). The cast's colours and pictures are set again at
render time, so one export serves every fight.
"""
import json, sys
from pathlib import Path
import numpy as np
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
FIGHT, OUT = Path(argv[0]).resolve(), Path(argv[1]).resolve()
VF = FIGHT.parent.parent
sys.path.insert(0, str(VF / "blender"))
import arena as A

data = json.loads((FIGHT / "fight.json").read_text(encoding="utf-8"))
ar = A.Arena(data["cast"], scale=50, pic_dir=FIGHT / "pics")


def val(sock):
    v = sock.default_value
    return list(v) if hasattr(v, "__len__") else float(v)


def material(m):
    nt = m.node_tree
    d = dict(blended=m.surface_render_method == "BLENDED")
    if m.get("glow"):
        em = next(n for n in nt.nodes if n.type == "EMISSION")
        mix = next(n for n in nt.nodes if n.type == "MIX_SHADER")
        d.update(kind="glow", emit=val(em.inputs["Color"])[:3], strength=val(em.inputs["Strength"]),
                 alpha=val(mix.inputs[0]))
        return d
    p = nt.nodes["Principled BSDF"]
    d.update(kind="ball" if "col" in m else "floor" if m.name == "floor" else "basic")
    for k, name in (("base", "Base Color"), ("rough", "Roughness"), ("metal", "Metallic"), ("coat", "Coat Weight"),
                    ("coat_rough", "Coat Roughness"), ("spec", "Specular IOR Level"), ("emit", "Emission Color"),
                    ("strength", "Emission Strength"), ("alpha", "Alpha")):
        s = p.inputs[name]
        if not s.is_linked:
            v = val(s)
            d[k] = v[:3] if isinstance(v, list) else v
    return d


def triangles(me):
    me.calc_loop_triangles()
    n = len(me.loop_triangles)
    loops = np.zeros(n * 3, "i4"); me.loop_triangles.foreach_get("loops", loops)
    vidx = np.zeros(len(me.loops), "i4"); me.loops.foreach_get("vertex_index", vidx)
    co = np.zeros(len(me.vertices) * 3, "f4"); me.vertices.foreach_get("co", co)
    nrm = np.array([tuple(c.vector) for c in me.corner_normals], "f4")
    mi = np.zeros(n, "i4"); me.loop_triangles.foreach_get("material_index", mi)
    return co.reshape(-1, 3)[vidx[loops]], nrm[loops], mi


hidden = set(o.name for o in ar.coll_hidden.objects)
objects, meshes, mats, lights, arrays = [], set(), {}, [], {}
for o in bpy.data.objects:
    if o.name in hidden:
        continue
    if o.type == "LIGHT":
        L = o.data
        d = dict(name=o.name, type=L.type, energy=L.energy, col=list(L.color), spec=L.specular_factor,
                 loc=list(o.location), dir=list(-o.matrix_world.to_3x3().col[2]))
        if L.type == "AREA": d["size"] = L.size
        if L.type == "SPOT": d.update(spot_size=L.spot_size, spot_blend=L.spot_blend)
        lights.append(d)
        continue
    if o.type != "MESH":
        continue
    slots = [s.material.name for s in o.material_slots]
    for s in o.material_slots: mats[s.material.name] = material(s.material)
    objects.append(dict(name=o.name, mesh=o.data.name, mats=slots, loc=list(o.location), rot=list(o.rotation_euler),
                        scale=list(o.scale), hide=o.hide_render))
    meshes.add(o.data.name)
for name in meshes:
    arrays[name + ":pos"], arrays[name + ":nrm"], arrays[name + ":mi"] = triangles(bpy.data.meshes[name])

nm = lambda x: x.name
pools = dict(
    balls=[{k: v.name for k, v in b.items()} for b in ar.balls],
    held=[{k: v.name for k, v in d.items()} for d in ar.held],
    pick=[{k: v.name for k, v in d.items()} for d in ar.pick],
    drop=[{k: v.name for k, v in d.items()} for d in ar.drop],
    bombs=list(map(nm, ar.bombs)), pads=list(map(nm, ar.pads)), danger=list(map(nm, ar.danger)),
    bullets=list(map(nm, ar.bullets)), rockets=list(map(nm, ar.rockets)), sights=list(map(nm, ar.sights)),
    shards=[list(map(nm, s)) for s in ar.shards],
    fx={k: [[o.name, m.name] for o, m in pool] for k, pool in ar.fx.items()},
    bullet_m={k: m.name for k, m in ar.bullet_m.items()},
    danger_m=ar.danger_m.name, pad_m=ar.pad_m.name, red_glow=ar.red_glow.name, neon_m=ar.neon_m.name,
    neon2_m=ar.neon2_m.name)
for m in bpy.data.materials:
    if m.use_nodes and m.name not in mats: mats[m.name] = material(m)
w = bpy.context.scene.world.node_tree.nodes["Background"]
info = dict(objects=objects, materials=mats, lights=lights, pools=pools,
            world=dict(col=val(w.inputs["Color"])[:3], strength=val(w.inputs["Strength"])))
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, **arrays)
OUT.with_suffix(".json").write_text(json.dumps(info, indent=0))
print(f"exported {len(objects)} objects, {len(meshes)} meshes, {len(mats)} materials, {len(lights)} lights -> {OUT}")
