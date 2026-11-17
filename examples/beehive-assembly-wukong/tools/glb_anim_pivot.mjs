// glb_anim_pivot.mjs — diagnose an animated GLB: does the HY-Motion motion tumble/invert
// around a misplaced root? Reports (per animation):
//   - how many rotation/translation channels resolve to real GLB nodes
//   - the deliberately animated root/hips Y-translation range (float source)
//   - the root/hips bone's PITCH (full-body tumble) sampled over the clip
// Usage: node tools/glb_anim_pivot.mjs <file.glb>
import fs from 'node:fs';
const p = process.argv[2];
if (!p) { console.log('usage: node glb_anim_pivot.mjs <file.glb>'); process.exit(0); }
const b = fs.readFileSync(p);
let json = null, off = 12, total = b.readUInt32LE(8);
while (off < total) { const cl = b.readUInt32LE(off); const ct = b.toString('utf8',off+4,off+8); off+=8; if(ct==='JSON'){ try{json=JSON.parse(b.toString('utf8',off,off+cl));}catch{} } off+=cl; }
const g = json || {};
console.log(`\n[${p}] ${(b.length/1048576).toFixed(2)}MB`);
const nodes = g.nodes || [];
const accessors = g.accessors || [];
const bufferViews = g.bufferViews || [];
let blob = null;
{ let o=12; while(o<total){ const cl=b.readUInt32LE(o); const ct=b.toString('utf8',o+4,o+8); o+=8; if(ct==='BIN'){ blob=b.subarray(o,o+cl); break; } o+=cl; } }
function readAcc(ai){
  if(ai==null||!accessors[ai])return null;
  const a=accessors[ai], bv=bufferViews[a.bufferView];
  if(!bv || !blob) return null;
  const base=bv.byteOffset||0;
  const cnt=a.count||0, comps=(a.type==='VEC3'?3:a.type==='VEC4'?4:1);
  return new Float32Array(blob.buffer, blob.byteOffset+base, cnt*comps);
}
for (let si=0; si<(g.animations||[]).length; si++){
  const an=g.animations[si];
  const samplers=an.samplers||[], channels=an.channels||[];
  console.log(`\n=== animation[${si}]: ${an.name} — ${samplers.length} samplers, ${channels.length} channels ===`);
  const nodeNames=new Set(nodes.map(n=>n.name));
  let rotR=0, rotU=0, traR=0, traU=0; const uni=[]; const rootY=[];
  // root pitch samples
  let rootPitch=[];
  for(const ch of channels){
    const path=ch.target.path, node=nodes[ch.target.node], nm=node?node.name:('node'+ch.target.node);
    const ok=!!(nm && nodeNames.has(nm) && node);
    if(path==='rotation'){ ok?rotR++:rotU++; if(!ok&&uni.length<8) uni.push('rot:'+nm);
      if(ok && /root|hips|pelvis|DEF-root|spine$/i.test(nm)){
        const vals=readAcc(samplers[ch.sampler].output); if(!vals){continue;}
        // sample pitch at ~5 spread frames
        const n=vals.length/4; const fs=[0,Math.floor(n/4),Math.floor(n/2),Math.floor(3*n/4),n-1];
        const times=(readAcc(samplers[ch.sampler].input));
        const eu=[]; for(const fi of fs){ const io=fi*4; const x=vals[io],y=vals[io+1],z=vals[io+2],w=vals[io+3];
          const sp=2*(w*y-z*x); let pitch=Math.asin(Math.max(-1,Math.min(1,sp))); pitch=pitch*180/Math.PI;
          eu.push(`${times?(+times[fi].toFixed(2))+'s':fi+'f'}=${pitch.toFixed(1)}°`); }
        if(!rootPitch.length) rootPitch.push(nm+': '+eu.join(', '));
      }
    }
    else if(path==='translation'){ ok?traR:traU++; if(!ok&&uni.length<8) uni.push('trans:'+nm);
      if(ok && /root|hips|pelvis|spine$/i.test(nm)){
        const vals=readAcc(samplers[ch.sampler].output);
        if(vals){ let mn=Infinity,mx=-Infinity; const n=vals.length/3; for(let i=0;i<n;i++){const y=vals[i*3+1];if(y<mn)mn=y;if(y>mx)mx=y;} rootY.push(`${nm} y[${mn.toFixed(3)}..${mx.toFixed(3)}]Δ${(mx-mn).toFixed(3)}`); }
      }
    }
  }
  console.log(`  ROT ${rotR} resolved / ${rotU} UNRES, TRAN ${traR} resolved / ${traU} UNRES`);
  if(uni.length) console.log('  unresolved samples:', uni.join(', '));
  if(rootPitch.length) console.log('  ROOT PITCH over clip (full-body tumble signal):', rootPitch.join(' | '));
  if(rootY.length) console.log('  ROOT/hips Y translation:', rootY.join(' | '));
}
if(g.skins && g.skins[0]){ const jn=g.skins[0].joints||[]; console.log('\nskeleton joints:', jn.length, '- first:', nodes.filter((n,i)=>jn.includes(i)).slice(0,8).map(n=>n.name).join(', ')); }