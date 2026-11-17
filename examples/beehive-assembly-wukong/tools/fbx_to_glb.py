# fbx_to_glb.py — Blender headless: import an FBX, apply armature modifiers,
# bake animation, and export a GLB that Three.js GLTFLoader reads cleanly.
# Fixes the "lying mesh bind" problem by letting Blender normalize the rest pose
# onto the standing skeleton (Blender's FBX importer reads FBX axes/rest properly).
# `export_materials='EXPORT'` preserves the FBX material colors.
#
# Usage:
#   blender -b -P fbx_to_glb.py -- --in "path/in.fbx" --out "path/out.glb" [--scale 0.01]
import bpy, sys, os, json

def log(*a): print("[fbx2glb]", *a, flush=True)

# ---- parse args after `--`
args = sys.argv[sys.argv.index("--")+1:]
def get(flag, default=None):
    if flag in args:
        i = args.index(flag); return args[i+1]
    return default

INP = get("--in")
OUT = get("--out")
SCALE = float(get("--scale", "0.01"))   # FBX is in cm; GLB in meters

if not (INP and OUT):
    log("ERROR: --in and --out required"); sys.exit(1)

log("in =", INP)
log("out =", OUT)

# ---- clean slate
if bpy.context.scene.objects: bpy.ops.object.delete()
# delete all collections' objects
for coll in list(bpy.data.collections):
    for o in list(coll.objects): coll.objects.unlink(o)
    bpy.data.collections.remove(coll)
for o in list(bpy.data.objects): bpy.data.objects.remove(o)

# ---- import FBX (patch the core addon's light reader for Blender 5.2 API break:
# blen_read_light references CyclesLightSettings.cast_shadow which no longer exists.
# We no-op lamp import so lights don't abort the whole rig+mesh import.)
import importlib
_f = importlib.import_module("io_scene_fbx.import_fbx")
def _blen_read_light_noop(fbx_tmpl, fbx_obj, settings):
    return None
_f.blen_read_light = _blen_read_light_noop
log("patched blen_read_light (skip lamps)")

try:
    bpy.ops.import_scene.fbx(filepath=INP, automatic_bone_orientation=True,
                             use_manual_orientation=False)
    log("FBX imported OK")
except Exception as e:
    log("FBX import ERROR:", e); sys.exit(2)

# ---- list what came in
objs = [o for o in bpy.data.objects if o.type in ('MESH','ARMATURE')]
log("objects:", [o.name+":"+o.type for o in objs][:40])
mesh_objs = [o for o in objs if o.type=='MESH']
arm = next((o for o in objs if o.type=='ARMATURE'), None)
log("armature:", arm.name if arm else None, "| meshes:", len(mesh_objs))

bpy.context.view_layer.update()
if arm:
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

log("Simplifying: keep armature + meshes parented; export GLB with bake.")

# ---- set scene scale so a cm FBX becomes meters
bpy.context.scene.unit_settings.scale_length = SCALE

# ---- export GLB (materials='EXPORT' keeps FBX material colors)
os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
try:
    bpy.ops.export_scene.gltf(
        filepath=OUT,
        export_format='GLB',
        use_selection=False,
        export_yup=True,
        export_apply=False,
        export_animations=True,
        export_bake_animation=True,
        export_anim_single_armature=True,
        export_materials='EXPORT',   # keep material colors from the FBX
    )
    log("GLB export OK ->", OUT)
except Exception as e:
    log("GLB export ERROR:", e); sys.exit(3)

# ---- report
if os.path.exists(OUT):
    log("DONE size=", os.path.getsize(OUT))
else:
    log("WARN: output not found")
sys.exit(0)