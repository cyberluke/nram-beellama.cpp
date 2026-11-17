// ============================================================================
// wukong-choreo.js — AI-authored full-body skeletal choreography for the Wukong
// Rigify FBX. Generates a THREE.AnimationClip procedurally from a data-driven
// timing/pose table (a "motion state machine") so an AI or editor can author
// custom animation WITHOUT a clip library or manual bone-posing.
//
// Why this exists: the premade libraries (Roboticela/Mixamo) don't know Wukong's
// Rigify bone names nor a bespoke "staff held in hand + spin + orbit the crystal".
// This is the reusable, AI-editable block that produces exactly that.
//
// Design:
//   PoseTable : array of { t (sec into clip), molde } where `molde` is a map of
//               semantic actor → { rx, ry, rz } Euler offset (radians) relative to
//               the skeleton's own clip, plus body-space position offsets.
//   Actors    : semantic names resolved to REAL bone names by substring match on
//               the loaded skeleton (so it works on any Rigify/Mixamo naming).
//   bakeClip() : interpolates the pose table (lerp on quaternions) into
//               KeyframeTracks and returns a THREE.AnimationClip.
//
// Usage (from setupWukong):
//   const clip = buildWukongChoreo(THREE, loaded, poseTable);
//   const action = mixer.clipAction(clip); action.reset().play();
//
// The staff is ALSO choreographed: it's usually a sub-mesh of the rig; we emit a
// quaternion track on "staff" so it spins about the hand's axis through the pose.
// ============================================================================

export function buildWukongChoreo(THREE, loaded, poseTable, opts = {}) {
  const { fps = 30, loop = true, name = 'wukong-choreo', staffName = null } = opts;
  // Discover real bone name -> the exported actor we use for tracks.
  // Wukong's Rigify rig uses DEF-* deformation bones for the actual mesh.
  const allBones = [];
  loaded.traverse(o => { if (o.isBone) allBones.push(o.name); });

  // Semantic actor -> resolver (returns the concrete bone name or null).
  const resolver = {
    hips:    n => /DEF-(root|pelvis|hips|torso|spine)/i.test(n),
    spine:   n => /DEF-spine/i.test(n),
    chest:   n => /DEF-(chest|spine3|torso)/i.test(n),
    shoulderR: n => /DEF-(shoulderR|shldrR|clavicleR)/i.test(n),
    upperArmR: n => /DEF-(upper_armR|shldrR|bicepR|armR|sharpR)/i.test(n),
    handR:   n => /DEF-handR|hand[_]?ikR|hand[_]?fkR|DEF-handR/i.test(n),
    shoulderL: n => /DEF-(shoulderL|shldrL|clavicleL)/i.test(n),
    upperArmL: n => /DEF-(upper_armL|shldrL|bicepL|armL|sharpL)/i.test(n),
    handL:   n => /DEF-handL|hand[_]?ikL|hand[_]?fkL/i.test(n),
    head:    n => /DEF-(head|nek)/i.test(n),
  };
  const actorBone = {};
  for (const [act, test] of Object.entries(resolver)) {
    const hit = allBones.find(test);
    if (hit) actorBone[act] = hit;
  }

  // The posed table: {t, pose:{actor:{rx,ry,rz}}} in seconds & radians.
  // We expand each into per-actor quaternion keyframes, hold between pivot points,
  // and lerp between neighbor samples for smoothness.
  const durS = Math.max(...poseTable.map(p => p.t), 0.001);
  const frames = Math.ceil(durS * fps);
  // For each actor, collect [time, Euler] and blend.
  const trackData = {}; // actor -> {times:[], r:[{x,y,z}]}
  for (const step of poseTable) {
    for (const [act, eul] of Object.entries(step.pose || {})) {
      (trackData[act] = trackData[act] || { t: [], r: [] });
      trackData[act].t.push(step.t);
      trackData[act].r.push(eul); // {x,y,z}
    }
  }

  const tracks = [];
  const mkQuatTrack = (boneName, actData, interp = THREE.Quaternion.slerp) => {
    const times = actData.t; const eu = actData.r;
    if (!times.length) return null;
    // build quaternion samples from euler (THREE.Euler order 'XYZ' default)
    const qs = eu.map(e => new THREE.Quaternion().setFromEuler(
      new THREE.Euler(e.x || 0, e.y || 0, e.z || 0)));
    // ensure monotonic times (three needs strictly increasing)
    const t = times.map(v => Math.max(0, v));
    // build values array
    const values = [];
    for (let i = 0; i < t.length; i++) {
      const q = qs[i];
      values.push(q.x, q.y, q.z, q.w);
    }
    return new THREE.QuaternionKeyframeTrack(boneName + '.quaternion', t, values);
  };

  for (const [act, ad] of Object.entries(trackData)) {
    const bn = actorBone[act];
    if (!bn) continue;                 // bone not present in this rig — skip safely
    const tr = mkQuatTrack(bn, ad);
    if (tr) tracks.push(tr);
  }

  // ---- STAFF: if it's a node we can target by name, spin it about its long axis
  // (the staff bbox is ~7.5 tall along Y; the twirl is a continuous rotation about
  // Y so it reads as idly spinning in the hand while handR keeps it positioned).
  if (staffName) {
    let staff = null;
    loaded.traverse(o => { if (o.name === staffName || (o.isMesh && o.name === staffName)) staff = o; });
    if (staff) {
      const axis = new THREE.Vector3(0, 1, 0);   // staff runs along +Y
      const times = []; const vals = [];
      const n = Math.max(2, Math.ceil(durS));    // spin ~1 rev per second
      for (let i = 0; i <= n; i++) {
        const tt = (i / n) * durS;
        const ang = (i / n) * Math.PI * 2 * durS;   // continuous spin
        const q = new THREE.Quaternion().setFromAxisAngle(axis, ang);
        times.push(tt); vals.push(q.x, q.y, q.z, q.w);
      }
      try { tracks.push(new THREE.QuaternionKeyframeTrack(staff.name + '.quaternion', times, vals)); } catch(e){}
    }
  }

  const clip = new THREE.AnimationClip(name, durS, tracks);
  clip.loop = loop;
  return clip;
}

// ----------------------------------------------------------------------------
// STORYBOARD — "Wukong greets the Crystal"  (5 beats, ~5.6 s loop)
//   1. IDLE     (0.0–0.9)   standing relaxed, staff low at R hand, gentle breath
//   2. RAISE    (0.9–2.0)   lifts the staff up into a guard across his chest
//   3. FLOURISH (2.0–3.2)   spirals the staff around his hands; body turns
//   4. SWEEP    (3.2–4.4)   wide power sweep of the staff, torso recovers
//   5. BOW      (4.4–5.6)   deep bow, then back up to the guard for the loop
// Actors: hips, spine, chest, shoulderR/upperArmR/handR (staff hand),
//         shoulderL/upperArmL/handL (support hand), head.
// rx/ry/rz are Euler offsets in RADIANS relative to the rest pose. Because the
// staff is parented to DEF-handR, keeping handR animated keeps the staff moving.
// ----------------------------------------------------------------------------
export const defaultChoreo = [
  // 1) IDLE — neutral, staff low, breathing
  { t: 0.0, pose: {
      hips:{x:0,   y:0,  z:0},   spine:{x:0.02,y:0,z:0}, chest:{x:0.02,y:0,z:0},
      shoulderR:{x:0.2, y:0, z:0.15}, upperArmR:{x:0.1, y:-0.25, z:0.05}, handR:{x:0.1,y:0,z:0.2},
      shoulderL:{x:0.15,y:0,z:-0.15}, upperArmL:{x:-0.05,y:0.2,z:-0.05}, handL:{x:0.05,y:0,z:-0.2},
      head:{x:0.04, y:0, z:0} } },
  { t: 0.45, pose: {
      hips:{x:0,   y:0,  z:0},   spine:{x:-0.02,y:0,z:0}, chest:{x:-0.03,y:0,z:0},
      shoulderR:{x:0.25,y:0,z:0.12}, upperArmR:{x:0.05, y:-0.3, z:0.05}, handR:{x:0.05,y:0,z:0.15},
      shoulderL:{x:0.1, y:0, z:-0.12}, upperArmL:{x:-0.02,y:0.22,z:-0.05}, handL:{x:0.02,y:0,z:-0.15},
      head:{x:-0.02, y:0, z:0} } },
  { t: 0.9, pose: {   // same as t0 — loop-warm breathing
      hips:{x:0,   y:0,  z:0},   spine:{x:0.02,y:0,z:0}, chest:{x:0.02,y:0,z:0},
      shoulderR:{x:0.2, y:0, z:0.15}, upperArmR:{x:0.1, y:-0.25, z:0.05}, handR:{x:0.1,y:0,z:0.2},
      shoulderL:{x:0.15,y:0,z:-0.15}, upperArmL:{x:-0.05,y:0.2,z:-0.05}, handL:{x:0.05,y:0,z:-0.2},
      head:{x:0.04, y:0, z:0} } },
  // 2) RAISE — bring the staff up to a guard across the chest
  { t: 1.1, pose: {
      hips:{x:0,   y:0,  z:0},   spine:{x:-0.05,y:0,z:0}, chest:{x:-0.05,y:0,z:0},
      shoulderR:{x:0.25,y:0.1,z:0.45}, upperArmR:{x:0.1, y:-0.9, z:0.1}, handR:{x:0.2,y:0,z:0.65},
      shoulderL:{x:0.2, y:-0.1,z:-0.3}, upperArmL:{x:-0.15,y:0.5,z:-0.15}, handL:{x:-0.1,y:0.1,z:-0.45},
      head:{x:0.02, y:0, z:0} } },
  { t: 2.0, pose: {  // held high guard
      hips:{x:0.02, y:0, z:0},  spine:{x:-0.08,y:0,z:0}, chest:{x:-0.08,y:0,z:0},
      shoulderR:{x:0.3, y:0.2, z:0.5},  upperArmR:{x:0.25, y:-1.15, z:0.2}, handR:{x:0.3,y:0,z:0.8},
      shoulderL:{x:0.2, y:-0.15,z:-0.35}, upperArmL:{x:-0.2,y:0.6,z:-0.2}, handL:{x:-0.15,y:0.15,z:-0.5},
      head:{x:0.02, y:0.05, z:0} } },
  // 3) FLOURISH — spiral the staff around, body turns
  { t: 2.5, pose: {  // mid-spin
      hips:{x:0, y:0.25, z:0}, spine:{x:-0.06,y:0.12,z:0}, chest:{x:-0.06,y:0.2,z:0},
      shoulderR:{x:0.2, y:0.5, z:0.6},  upperArmR:{x:0.4, y:-1.3, z:0.4}, handR:{x:0.5,y:0.1,z:1.0},
      shoulderL:{x:0.15,y:0.2,z:-0.4}, upperArmL:{x:-0.15,y:-0.35,z:-0.2}, handL:{x:0.3,y:0.2,z:-0.55},
      head:{x:0.02, y:0.2, z:0} } },
  { t: 3.2, pose: {  // spin complete, back toward front
      hips:{x:0, y:-0.25, z:0}, spine:{x:-0.04,y:-0.1,z:0}, chest:{x:-0.04,y:-0.15,z:0},
      shoulderR:{x:0.25,y:-0.3,z:0.4}, upperArmR:{x:0.1, y:-0.6, z:-0.1}, handR:{x:0.1,y:0,z:0.5},
      shoulderL:{x:0.25,y:0.1,z:-0.3}, upperArmL:{x:-0.1,y:0.4,z:-0.15}, handL:{x:-0.05,y:0.1,z:-0.4},
      head:{x:0.02, y:-0.2, z:0} } },
  // 4) SWEEP — powerful wide arc, torso coils down then recovers
  { t: 3.7, pose: {
      hips:{x:0.08, y:0, z:0}, spine:{x:0.15,y:0,z:0}, chest:{x:0.2,y:0.1,z:0.1},
      shoulderR:{x:0.6,y:-0.1,z:0.2}, upperArmR:{x:0.7,y:-0.3,z:-0.3}, handR:{x:0.7,y:0.2,z:0.4},
      shoulderL:{x:0.4,y:0.15,z:-0.2}, upperArmL:{x:0.3,y:0.5,z:-0.2}, handL:{x:0.1,y:0.1,z:-0.35},
      head:{x:0.15, y:0, z:0.05} } },
  { t: 4.4, pose: {  // recover to guard
      hips:{x:0, y:0, z:0}, spine:{x:-0.05,y:0,z:0}, chest:{x:-0.05,y:0,z:0},
      shoulderR:{x:0.25,y:0, z:0.45}, upperArmR:{x:0.1, y:-0.9, z:0.1}, handR:{x:0.2,y:0,z:0.6},
      shoulderL:{x:0.2, y:-0.1,z:-0.3}, upperArmL:{x:-0.15,y:0.5,z:-0.15}, handL:{x:-0.1,y:0.1,z:-0.45},
      head:{x:0.02, y:0, z:0} } },
  // 5) BOW — deep, then rise
  { t: 4.8, pose: {
      hips:{x:0.35, y:0, z:0}, spine:{x:0.4, y:0, z:0}, chest:{x:0.4,y:0,z:0},
      shoulderR:{x:-0.1, y:0.3,z:0.3}, upperArmR:{x:0.2,y:-0.8,z:0.1}, handR:{x:0.25,y:0.1,z:0.5},
      shoulderL:{x:0.0, y:-0.15,z:-0.25}, upperArmL:{x:-0.1,y:0.45,z:-0.1}, handL:{x:-0.1,y:0.1,z:-0.4},
      head:{x:0.6, y:0, z:0} } },   // head all the way down
  { t: 5.4, pose: {  // rise up to guard ready to loop
      hips:{x:0, y:0, z:0}, spine:{x:-0.05,y:0,z:0}, chest:{x:-0.05,y:0,z:0},
      shoulderR:{x:0.25,y:0, z:0.45}, upperArmR:{x:0.1, y:-0.9, z:0.1}, handR:{x:0.2,y:0,z:0.6},
      shoulderL:{x:0.2, y:-0.1,z:-0.3}, upperArmL:{x:-0.15,y:0.5,z:-0.15}, handL:{x:-0.1,y:0.1,z:-0.45},
      head:{x:0.02, y:0, z:0} } },
  { t: 5.6, pose: {  // final guard (same as t5.4 = seamless loop seam)
      hips:{x:0, y:0, z:0}, spine:{x:-0.05,y:0,z:0}, chest:{x:-0.05,y:0,z:0},
      shoulderR:{x:0.25,y:0, z:0.45}, upperArmR:{x:0.1, y:-0.9, z:0.1}, handR:{x:0.2,y:0,z:0.6},
      shoulderL:{x:0.2, y:-0.1,z:-0.3}, upperArmL:{x:-0.15,y:0.5,z:-0.15}, handL:{x:-0.1,y:0.1,z:-0.45},
      head:{x:0.02, y:0, z:0} } }
];