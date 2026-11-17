// glb_info.mjs — inspect a GLB: JSON chunk stats (nodes/skins/meshes/materials/animations),
// first material's baseColorFactor, sample bone names, file size.
// Usage: node tools/glb_info.mjs <file.glb> [<file2.glb> ...]
import fs from 'node:fs';

for (const p of process.argv.slice(2)) {
  if (!fs.existsSync(p)) { console.log(p, ': missing'); continue; }
  const b = fs.readFileSync(p);
  const sizeMB = (b.length / 1048576).toFixed(2);
  const magic = b.toString('utf8', 0, 4);
  const version = b.readUInt32LE(4);
  const totalLen = b.readUInt32LE(8);
  let json = null, off = 12;
  while (off < totalLen) {
    const cl = b.readUInt32LE(off);
    const ct = b.toString('utf8', off + 4, off + 8);
    off += 8;
    if (ct === 'JSON') { try { json = JSON.parse(b.toString('utf8', off, off + cl)); } catch {} }
    off += cl;
  }
  const g = json || {};
  const mat = (g.materials || []);
  const namedColors = mat.map(m => ({
    name: m.name || 'unnamed',
    base: m.pbrMetallicRoughness && m.pbrMetallicRoughness.baseColorFactor,
  }));
  const boneNames = (g.nodes || []).map(n=>n.name||'').filter(n=>/DEF-|mixamorig|hand|spine|pelvis|head/i.test(n)).slice(0,14);
  console.log(`\n[${p}] ${sizeMB} MB · ${magic} v${version}`);
  console.log(`  nodes=${(g.nodes||[]).length} skins=${(g.skins||[]).length} meshes=${(g.meshes||[]).length} anims=${(g.animations||[]).length}`);
  console.log(`  materials=${mat.length} → colors: ${namedColors.map(m=>m.name+'='+(m.base?m.base.slice(0,3).join('/'):'?')).join(', ') || '(no materials = GREY)'}`);
  console.log(`  bones(sample): ${boneNames.join(', ') || '(none)'}`);
}