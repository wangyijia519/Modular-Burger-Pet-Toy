// ============================================================================
// sdf_mesh.js
// Analytical 3D Signed Distance Functions (SDF) & Isosurface Mesh Generator
// Target Dimensions: Outer Diameter = 110 mm (R = 55 mm = 1.00 unit)
//                    Stacked Height = 112 mm (H = 2.0364 units)
// ============================================================================

export const MM_PER_UNIT = 55.0; // 1.0 world unit = 55.0 mm (Radius = 1.0 -> Diameter = 110.0 mm)

// Smooth boolean primitives
function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

function smax(a, b, k) {
  return -smin(-a, -b, k);
}

function clamp(x, minVal, maxVal) {
  return x < minVal ? minVal : (x > maxVal ? maxVal : x);
}

function smoothstep(e0, e1, x) {
  const t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// Precompute teardrop sesame seed indent frames on Top Shell (Layer 0) and Bottom Shell (Layer 4)
const TOP_SESAME_SEEDS = [];
const BOT_SESAME_SEEDS = [];

(function initSesameSeeds() {
  // Top Shell: 3 concentric rings matching reference photo
  const rings = [
    { count: 5, phi: 0.34, phase: 0.25, tilt: 0.16 },
    { count: 8, phi: 0.68, phase: 0.08, tilt: -0.20 },
    { count: 9, phi: 1.04, phase: 0.45, tilt: 0.14 }
  ];
  for (let rIdx = 0; rIdx < rings.length; rIdx++) {
    const ring = rings[rIdx];
    for (let i = 0; i < ring.count; i++) {
      const theta = (i / ring.count) * Math.PI * 2.0 + ring.phase + ((i % 2) * 0.07);
      const phi = ring.phi + ((i % 3) - 1) * 0.03;
      const sinP = Math.sin(phi);
      const cosP = Math.cos(phi);
      const nx = sinP * Math.cos(theta);
      const ny = cosP;
      const nz = sinP * Math.sin(theta);

      // Meridian tangent (pointing from rim toward pole)
      let tx = -cosP * Math.cos(theta);
      let ty = sinP;
      let tz = -cosP * Math.sin(theta);
      // Azimuth tangent
      let bx = -Math.sin(theta);
      let by = 0.0;
      let bz = Math.cos(theta);

      // Slight organic angular tilt matching photo
      const ang = ring.tilt + ((i % 2 === 0) ? 0.12 : -0.12);
      const ca = Math.cos(ang), sa = Math.sin(ang);
      const u1x = ca * tx + sa * bx;
      const u1y = ca * ty + sa * by;
      const u1z = ca * tz + sa * bz;
      const u2x = -sa * tx + ca * bx;
      const u2y = -sa * ty + ca * by;
      const u2z = -sa * tz + ca * bz;

      TOP_SESAME_SEEDS.push({
        nx, ny, nz,
        u1x, u1y, u1z,
        u2x, u2y, u2z
      });
    }
  }

  // Bottom Shell: 2 rings of teardrop indents on lower curved bowl wall
  const botRings = [
    { count: 7, phi: 1.96, phase: 0.32, tilt: 0.20 },
    { count: 6, phi: 2.30, phase: 0.78, tilt: -0.16 }
  ];
  for (let rIdx = 0; rIdx < botRings.length; rIdx++) {
    const ring = botRings[rIdx];
    for (let i = 0; i < ring.count; i++) {
      const theta = (i / ring.count) * Math.PI * 2.0 + ring.phase;
      const phi = ring.phi + ((i % 2) * 0.04);
      const sinP = Math.sin(phi);
      const cosP = Math.cos(phi);
      const nx = sinP * Math.cos(theta);
      const ny = cosP; // negative (lower hemisphere)
      const nz = sinP * Math.sin(theta);

      let tx = -cosP * Math.cos(theta);
      let ty = sinP;
      let tz = -cosP * Math.sin(theta);
      let bx = -Math.sin(theta);
      let by = 0.0;
      let bz = Math.cos(theta);

      const ang = ring.tilt;
      const ca = Math.cos(ang), sa = Math.sin(ang);
      BOT_SESAME_SEEDS.push({
        nx, ny, nz,
        u1x: ca * tx + sa * bx,
        u1y: ca * ty + sa * by,
        u1z: ca * tz + sa * bz,
        u2x: -sa * tx + ca * bx,
        u2y: -sa * ty + ca * by,
        u2z: -sa * tz + ca * bz
      });
    }
  }
})();

export { TOP_SESAME_SEEDS, BOT_SESAME_SEEDS };

// Evaluate teardrop sesame indent depth & mask at a point on a unit-radius shell
export function evalSesameIndents(px, py, pz, seeds, isTop) {
  const r = Math.sqrt(px * px + py * py + pz * pz) + 1e-6;
  const ux = px / r, uy = py / r, uz = pz / r;
  let maxIndent = 0.0;

  // Teardrop semi-axes in radians/world-units (R ~ 1.0)
  const lenA = isTop ? 0.095 : 0.085; // ~5.2 mm half-length
  const widB = isTop ? 0.052 : 0.046; // ~2.9 mm half-width

  for (let i = 0; i < seeds.length; i++) {
    const s = seeds[i];
    const dotN = ux * s.nx + uy * s.ny + uz * s.nz;
    if (dotN < 0.955) continue; // Fast spherical cap cull

    // Local tangent coordinates: v along major axis (tip at +lenA), u across width
    const v = px * s.u1x + py * s.u1y + pz * s.u1z;
    const u = px * s.u2x + py * s.u2y + pz * s.u2z;

    const vn = clamp(v / lenA, -1.2, 1.2);
    // Asymmetric teardrop taper: narrower near tip (+1), plump near base (-1)
    const taper = 1.0 - 0.38 * vn;
    const effW = Math.max(0.014, widB * taper);
    const un = u / effW;
    const q2 = un * un + vn * vn;
    if (q2 < 1.0) {
      const falloff = 1.0 - q2;
      const profile = falloff * falloff;
      if (profile > maxIndent) maxIndent = profile;
    }
  }
  return maxIndent;
}

// ============================================================================
// LAYER 0: Cream-Yellow Hemispherical Top Shell with Teardrop Sesame Indents
// Local coordinate frame: base rim at y = 0.0, dome crown at y = +0.84
// ============================================================================
export function sdfLayer0(x, y, z) {
  const r2 = x * x + z * z;

  // Outer oblate hemisphere centered at y = -0.04 so base at y=0 has R=1.00 and apex is at y=0.84
  const cy = y + 0.04;
  const sy = cy / 0.88;
  const dSphereOut = (Math.sqrt(r2 + sy * sy) - 1.00) * 0.90;

  // Flat horizontal base plane at y = 0.0 with generous rounded rim fillet
  const dBaseCut = -y;
  let dOuter = smax(dSphereOut, dBaseCut, 0.052);

  // Inner hollow dome cavity (wall thickness = 0.135 units = 7.4 mm for clean smooth rim)
  const syIn = (y + 0.06) / 0.76;
  const dSphereIn = (Math.sqrt(r2 / (0.865 * 0.865) + syIn * syIn) - 1.00) * 0.80;
  let dShell = smax(dOuter, -dSphereIn, 0.045);

  // Carve sculptural teardrop sesame indents into outer surface
  if (y > 0.05 && Math.abs(dOuter) < 0.11 && dSphereOut > dSphereIn * 0.5) {
    const indent = evalSesameIndents(x, cy / 0.88, z, TOP_SESAME_SEEDS, true);
    if (indent > 0.0) {
      dShell += indent * 0.038;
    }
  }

  return dShell;
}

// ============================================================================
// LAYER 1: Matcha-Green Wavy Ruffled Snuffle Fabric Layer
// Double-tiered wavy ruffled felt skirt matching reference photo
// Local coordinate frame: y in [-0.18, +0.18], outer radius ~ 1.02
// ============================================================================
export function sdfLayer1(x, y, z) {
  const r = Math.sqrt(x * x + z * z);
  const theta = Math.atan2(z, x);

  // Radial flare factor: flat cohesive core inside r < 0.28, deep ruffles toward perimeter
  const flare = smoothstep(0.22, 0.90, r);
  const outerScallop1 = 0.985 + 0.040 * Math.cos(9.0 * theta + 0.2) + 0.015 * Math.sin(18.0 * theta);
  const outerScallop2 = 1.005 + 0.042 * Math.sin(9.0 * theta - 0.5) + 0.015 * Math.cos(18.0 * theta);

  // Upper wavy ruffled felt fold
  const wave1 = flare * (
    0.068 * Math.sin(9.0 * theta + 1.5 * r) +
    0.022 * Math.cos(18.0 * theta - 2.0 * r)
  );
  const yCenter1 = 0.048 + wave1;
  const halfThick = 0.034; // plush felt sheet thickness
  const dVert1 = (Math.abs(y - yCenter1) - halfThick) * 0.85;
  const dRad1 = smax(0.24 - r, r - outerScallop1, 0.04);
  const dTier1 = smax(dVert1, dRad1, 0.034);

  // Lower wavy ruffled felt fold (creates deep snuffle pockets between upper & lower folds)
  const wave2 = flare * (
    0.070 * Math.cos(9.0 * theta - 1.4 * r + 0.8) +
    0.022 * Math.sin(18.0 * theta + 1.8 * r)
  );
  const yCenter2 = -0.048 + wave2;
  const dVert2 = (Math.abs(y - yCenter2) - halfThick) * 0.85;
  const dRad2 = smax(0.24 - r, r - outerScallop2, 0.04);
  const dTier2 = smax(dVert2, dRad2, 0.034);

  return smin(dTier1, dTier2, 0.036);
}

// ============================================================================
// LAYER 2: Coral-Red Translucent Hollow TPE Chew Ring with Rounded Outer Bumps
// Compact toroidal chew ring with raised inner rim lip and 12 plump outer lobes
// matching the reference photo's exact silhouette!
// ============================================================================
export function sdfLayer2(x, y, z) {
  const r = Math.sqrt(x * x + z * z);
  const theta = Math.atan2(z, x);

  // 1. Toroidal base ring body (r in [0.58, 0.90], compact height ~ 0.28 units)
  const rMid = 0.74;
  const dr = r - rMid;
  // Raised inner collar rim at r ~ 0.635 (visible on top inner edge in reference photo)
  const innerRimLip = 0.036 * Math.exp(-(((r - 0.635) / 0.048) ** 2));
  const halfH = 0.088 + innerRimLip;

  const qR0 = Math.max(Math.abs(dr) - 0.105, 0.0);
  const qY0 = Math.max(Math.abs(y) - halfH, 0.0);
  const dCoreRing = Math.sqrt(qR0 * qR0 + qY0 * qY0) +
                    Math.min(Math.max(Math.abs(dr) - 0.105, Math.abs(y) - halfH), 0.0) - 0.048;

  // 2. 12 distinct, plump rounded outer cushion bumps around perimeter with clear vertical valleys
  const numBumps = 12.0;
  const sector = (Math.PI * 2.0) / numBumps;
  const halfSector = sector * 0.5;
  let dTheta = ((theta % sector) + sector) % sector;
  if (dTheta > halfSector) dTheta -= sector;

  // Arc-length coordinate along outer circumference
  const arcU = dTheta * 0.88;
  const bumpR = r - 0.855;
  const bumpY = y + 0.010;

  // Rounded cushion lobe per sector
  const bU = Math.max(Math.abs(arcU) - 0.082, 0.0);
  const bY = Math.max(Math.abs(bumpY) - 0.056, 0.0);
  const bR = Math.max(Math.abs(bumpR) - 0.038, 0.0);
  const dBump = Math.sqrt(bU * bU + bY * bY + bR * bR) +
                Math.min(Math.max(Math.abs(arcU) - 0.082, Math.max(Math.abs(bumpY) - 0.056, Math.abs(bumpR) - 0.038)), 0.0) - 0.068;

  // Carve subtle vertical indent grooves between the 12 bumps on the outer belt
  const grooveAngle = Math.abs(Math.abs(dTheta) - halfSector);
  const grooveMask = Math.exp(-((grooveAngle / 0.055) ** 2)) * smoothstep(0.80, 0.92, r);

  let dOuterRing = smin(dCoreRing, dBump, 0.032) + grooveMask * 0.010;

  // 3. Internal hollow toroidal channel inside the TPE ring (gives hollow squish + translucent wall)
  const dHollowR = r - 0.735;
  const dHollow = Math.sqrt(dHollowR * dHollowR + y * y * 1.85) - 0.048;

  return smax(dOuterRing, -dHollow, 0.022);
}

// ============================================================================
// LAYER 3: Warm-Orange Frisbee Disc Module with a Recessed Rim
// Local coordinate frame: y in [-0.11, +0.11], outer radius = 0.89 (98 mm dia)
// ============================================================================
export function sdfLayer3(x, y, z) {
  const r2 = x * x + z * z;
  const r = Math.sqrt(r2);

  // Rounded aerodynamic disc outer profile
  const dRad = r - 0.840;
  const dVert = Math.abs(y) - 0.054;
  const dBaseDisc = Math.sqrt(Math.max(dRad, 0.0) ** 2 + Math.max(dVert, 0.0) ** 2) +
                    Math.min(Math.max(dRad, dVert), 0.0) - 0.048;

  // Top circular recessed well with subtle concave dish curvature (eliminates flat-slice MC coplanarity)
  const dTopWellR = r - 0.680;
  const wellFloorY = 0.038 + 0.022 * r2;
  const dTopWellY = wellFloorY - y;
  const dTopWell = smax(dTopWellR, dTopWellY, 0.038);

  // Subtle underside flight grip dish (inside r < 0.61)
  const dBotWellR = r - 0.610;
  const botFloorY = -0.056 - 0.016 * r2;
  const dBotWellY = y - botFloorY;
  const dBotWell = smax(dBotWellR, dBotWellY, 0.038);

  let dDisc = smax(dBaseDisc, -dTopWell, 0.030);
  dDisc = smax(dDisc, -dBotWell, 0.030);
  return dDisc;
}

// ============================================================================
// LAYER 4: Cream-Yellow Bowl-Shaped Bottom Shell with a Semi-Circular Notch
// Local coordinate frame: top rim at y = 0.0, flat bottom base at y = -0.82
// ============================================================================
export function sdfLayer4(x, y, z) {
  const r2 = x * x + z * z;

  // Lower hemisphere outer wall centered at y = +0.04 so top rim at y=0 has R = 1.00
  const cy = y - 0.04;
  const sy = cy / 0.88;
  const dSphereOut = (Math.sqrt(r2 + sy * sy) - 1.00) * 0.90;

  // Top rim horizontal plane at y = 0.0 and flat bottom base plane at y = -0.82
  const dTopCut = y;
  const dBotFlat = -0.82 - y;
  let dOuter = smax(dSphereOut, dTopCut, 0.048);
  dOuter = smax(dOuter, dBotFlat, 0.068);

  // Inner hollow bowl cavity (wall thickness = 0.135 units = 7.4 mm for clean thick rim)
  const syIn = (y - 0.05) / 0.78;
  const dSphereIn = (Math.sqrt(r2 / (0.865 * 0.865) + syIn * syIn) - 1.00) * 0.80;
  const dInnerFloor = -0.69 - y;
  const dCavity = smax(dSphereIn, dInnerFloor, 0.060);

  let dBowl = smax(dOuter, -dCavity, 0.042);

  // Semi-circular notch on front-right upper rim (matching reference photo azimuth ~ +22 deg)
  const notchAngle = 0.36; // ~20.6 degrees right of front (+Z)
  const nx = Math.sin(notchAngle) * 0.930;
  const nz = Math.cos(notchAngle) * 0.930;
  const dx = x - nx;
  const dy = y - 0.015;
  const dz = z - nz;
  // Radial axis for cylindrical/spherical U-notch across the rim wall
  const radProj = dx * Math.sin(notchAngle) + dz * Math.cos(notchAngle);
  const tanProj = dx * Math.cos(notchAngle) - dz * Math.sin(notchAngle);
  const dNotchCross = Math.sqrt(tanProj * tanProj + dy * dy) - 0.135; // 7.4 mm notch radius
  const dNotch = smax(dNotchCross, Math.abs(radProj) - 0.24, 0.025);

  dBowl = smax(dBowl, -dNotch, 0.032);

  // Teardrop sesame indents on the lower outer bowl surface (matching reference photo)
  if (y < -0.10 && y > -0.72 && Math.abs(dOuter) < 0.10 && dSphereOut > dCavity * 0.5) {
    const indent = evalSesameIndents(x, cy / 0.88, z, BOT_SESAME_SEEDS, false);
    if (indent > 0.0) {
      dBowl += indent * 0.032;
    }
  }

  return dBowl;
}

// ============================================================================
// SNUFFLE PET TREATS (Layers 5, 6, 7): Freeze-Dried Cubes & Crunchy Star Kibble
// Designed to be tucked into the 9 wavy ruffle pockets of Layer 1 (Matcha Felt)
// ============================================================================

// Layer 5: Freeze-Dried Egg Yolk / Cheddar Cube (#F2A93B)
export function sdfTreatEggCube(x, y, z) {
  // Slightly pillowed freeze-dried cube (~8.2 mm x 6.8 mm x 8.2 mm)
  const r2 = x * x + z * z;
  const pillow = 0.006 * Math.cos(x * 28.0) * Math.cos(z * 28.0);
  const qx = Math.max(Math.abs(x) - 0.054, 0.0);
  const qy = Math.max(Math.abs(y) - (0.042 + pillow), 0.0);
  const qz = Math.max(Math.abs(z) - 0.054, 0.0);
  const dOut = Math.sqrt(qx * qx + qy * qy + qz * qz);
  const dIn = Math.min(Math.max(Math.abs(x) - 0.054, Math.max(Math.abs(y) - 0.042, Math.abs(z) - 0.054)), 0.0);
  return dOut + dIn - 0.022 + 0.002 * r2;
}

// Layer 6: Freeze-Dried Salmon / Meat Chunk (#DF6D53)
export function sdfTreatSalmonChunk(x, y, z) {
  // Organic beveled freeze-dried meat bite (~9.0 mm x 6.4 mm x 7.5 mm)
  const wave = 0.004 * Math.sin(x * 32.0 + z * 24.0);
  const qx = Math.max(Math.abs(x) - 0.062, 0.0);
  const qy = Math.max(Math.abs(y - wave) - 0.038, 0.0);
  const qz = Math.max(Math.abs(z) - 0.048, 0.0);
  const dOut = Math.sqrt(qx * qx + qy * qy + qz * qz);
  const dIn = Math.min(Math.max(Math.abs(x) - 0.062, Math.max(Math.abs(y - wave) - 0.038, Math.abs(z) - 0.048)), 0.0);
  return dOut + dIn - 0.020;
}

// Layer 7: Crunchy 5-Lobed Star / Flower Kibble Biscuit (#BD7438)
export function sdfTreatStarBiscuit(x, y, z) {
  const r = Math.sqrt(x * x + z * z);
  const theta = Math.atan2(z, x);
  const starR = 0.066 + 0.015 * Math.cos(5.0 * theta);
  const dRad = (r - starR) * 0.88;
  const dVert = Math.abs(y) - 0.032;
  const dBody = Math.sqrt(Math.max(dRad, 0.0) ** 2 + Math.max(dVert, 0.0) ** 2) +
                Math.min(Math.max(dRad, dVert), 0.0) - 0.016;
  // Cute central baked biscuit dimple on top & bottom faces
  const dDimple = Math.sqrt(r * r + (Math.abs(y) - 0.052) ** 2) - 0.026;
  return smax(dBody, -dDimple, 0.012);
}

export function evalLayerSDF(layerId, x, y, z) {
  switch (layerId) {
    case 0: return sdfLayer0(x, y, z);
    case 1: return sdfLayer1(x, y, z);
    case 2: return sdfLayer2(x, y, z);
    case 3: return sdfLayer3(x, y, z);
    case 4: return sdfLayer4(x, y, z);
    case 5: return sdfTreatEggCube(x, y, z);
    case 6: return sdfTreatSalmonChunk(x, y, z);
    case 7: return sdfTreatStarBiscuit(x, y, z);
    default: return 1.0;
  }
}

// Compute 4-point tetrahedral SDF gradient normal
export function evalLayerNormal(layerId, x, y, z, eps = 0.0025) {
  const d0 = evalLayerSDF(layerId, x + eps, y - eps, z - eps);
  const d1 = evalLayerSDF(layerId, x - eps, y - eps, z + eps);
  const d2 = evalLayerSDF(layerId, x - eps, y + eps, z - eps);
  const d3 = evalLayerSDF(layerId, x + eps, y + eps, z + eps);

  let gx = (d0 - d1 - d2 + d3);
  let gy = (-d0 - d1 + d2 + d3);
  let gz = (-d0 + d1 - d2 + d3);
  const len = Math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9;
  return [gx / len, gy / len, gz / len];
}

// Estimate local optical/structural thickness by marching inward along -normal
export function evalLayerThickness(layerId, x, y, z, nx, ny, nz) {
  let maxDepth = 0.0;
  // Probe at 6 steps inward up to 0.30 units (~16.5 mm)
  for (let i = 1; i <= 6; i++) {
    const t = i * 0.044;
    const d = evalLayerSDF(layerId, x - nx * t, y - ny * t, z - nz * t);
    if (d < 0.010) {
      maxDepth = t + Math.abs(d);
    }
  }
  return clamp(maxDepth / 0.24, 0.12, 1.0);
}

// ============================================================================
// Layer Configuration & Vertical Offsets Across the 3 Modes
// ============================================================================
export const LAYER_CONFIGS = [
  {
    id: 0,
    name: "Top Shell (Bun Crown)",
    shortName: "Top Shell",
    cnName: "上壳体",
    materialType: "Rigid-Flex Silicone (65A)",
    colorHex: "#EFE0B4",
    densityGPerCm3: 1.18,
    bbox: { minX: -1.06, maxX: 1.06, minY: -0.05, maxY: 0.89, minZ: -1.06, maxZ: 1.06 },
    gridRes: [60, 32, 60],
    hexRes: [8, 4, 8],
    // Vertical rest offset of local y=0 in each mode:
    // Burger Snuffle Mode: total height from -1.018 to +1.018 = 2.0364 units (112.0 mm)
    yBurger: 0.178,   // local apex +0.84 -> world +1.018
    ySphere: 0.012,   // closes flush onto bottom shell rim with 1.3 mm equatorial seam
    explodeRank: +2.0
  },
  {
    id: 1,
    name: "Snuffle Fabric Layer",
    shortName: "Snuffle Felt",
    cnName: "藏食布层",
    materialType: "Needle-Punched Felt (Matcha)",
    colorHex: "#6D8849",
    densityGPerCm3: 0.42,
    bbox: { minX: -1.08, maxX: 1.08, minY: -0.19, maxY: 0.19, minZ: -1.08, maxZ: 1.08 },
    gridRes: [62, 24, 62],
    hexRes: [10, 3, 10],
    yBurger: 0.115,
    ySphere: 0.0,
    explodeRank: +1.0
  },
  {
    id: 2,
    name: "Coral-Red TPE Chew Ring",
    shortName: "TPE Chew Ring",
    cnName: "珊瑚红软胶层",
    materialType: "Translucent Jelly TPE (22A)",
    colorHex: "#F05E46",
    densityGPerCm3: 0.96,
    bbox: { minX: -1.04, maxX: 1.04, minY: -0.18, maxY: 0.18, minZ: -1.04, maxZ: 1.04 },
    gridRes: [62, 24, 62],
    hexRes: [9, 3, 9],
    yBurger: -0.075,
    ySphere: 0.0,
    explodeRank: 0.0
  },
  {
    id: 3,
    name: "Warm-Orange Frisbee Module",
    shortName: "Frisbee Disc",
    cnName: "暖橙色飞盘模块",
    materialType: "Medium-Flex Elastomer (45A)",
    colorHex: "#EA8238",
    densityGPerCm3: 1.08,
    bbox: { minX: -0.94, maxX: 0.94, minY: -0.135, maxY: 0.125, minZ: -0.94, maxZ: 0.94 },
    gridRes: [58, 22, 58],
    hexRes: [8, 2, 8],
    yBurger: -0.225, // nests inside upper rim of Bottom Shell under the TPE Chew Ring
    ySphere: 0.0,
    explodeRank: -1.0
  },
  {
    id: 4,
    name: "Bottom Shell (Notched Bowl)",
    shortName: "Bottom Shell",
    cnName: "下壳体",
    materialType: "Rigid-Flex Silicone (65A)",
    colorHex: "#EFE0B4",
    densityGPerCm3: 1.18,
    bbox: { minX: -1.06, maxX: 1.06, minY: -0.87, maxY: 0.05, minZ: -1.06, maxZ: 1.06 },
    gridRes: [60, 32, 60],
    hexRes: [8, 4, 8],
    yBurger: -0.198,  // local bottom -0.82 -> world -1.018 (total span = 2.0364 = 112.0 mm)
    ySphere: -0.012,  // closes against top shell rim
    explodeRank: -2.0
  }
];

// ============================================================================
// Standard Marching Cubes Lookup Tables (Edge Mask & Triangle Table)
// ============================================================================
const MC_EDGE_TABLE = new Int32Array([
  0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
  0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
  0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
  0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
  0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
  0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
  0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
  0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
  0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
  0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
  0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
  0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
  0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
  0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
  0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
  0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
  0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
  0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
  0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
  0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
  0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
  0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
  0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
  0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
  0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
  0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
  0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
  0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
  0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
  0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
  0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
  0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
]);

const MC_TRI_TABLE = [
  [-1],
  [0, 8, 3, -1],
  [0, 1, 9, -1],
  [1, 8, 3, 9, 8, 1, -1],
  [1, 2, 10, -1],
  [0, 8, 3, 1, 2, 10, -1],
  [9, 2, 10, 0, 2, 9, -1],
  [2, 8, 3, 2, 10, 8, 10, 9, 8, -1],
  [3, 11, 2, -1],
  [0, 11, 2, 8, 11, 0, -1],
  [1, 9, 0, 2, 3, 11, -1],
  [1, 11, 2, 1, 9, 11, 9, 8, 11, -1],
  [3, 10, 1, 11, 10, 3, -1],
  [0, 10, 1, 0, 8, 10, 8, 11, 10, -1],
  [3, 9, 0, 3, 11, 9, 11, 10, 9, -1],
  [9, 8, 10, 10, 8, 11, -1],
  [4, 7, 8, -1],
  [4, 3, 0, 7, 3, 4, -1],
  [0, 1, 9, 8, 4, 7, -1],
  [4, 1, 9, 4, 7, 1, 7, 3, 1, -1],
  [1, 2, 10, 8, 4, 7, -1],
  [3, 4, 7, 3, 0, 4, 1, 2, 10, -1],
  [9, 2, 10, 9, 0, 2, 8, 4, 7, -1],
  [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1],
  [8, 4, 7, 3, 11, 2, -1],
  [11, 4, 7, 11, 2, 4, 2, 0, 4, -1],
  [9, 0, 1, 8, 4, 7, 2, 3, 11, -1],
  [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1],
  [3, 10, 1, 3, 11, 10, 7, 8, 4, -1],
  [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1],
  [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1],
  [4, 7, 11, 4, 11, 9, 9, 11, 10, -1],
  [9, 5, 4, -1],
  [9, 5, 4, 0, 8, 3, -1],
  [0, 5, 4, 1, 5, 0, -1],
  [8, 5, 4, 8, 3, 5, 3, 1, 5, -1],
  [1, 2, 10, 9, 5, 4, -1],
  [3, 0, 8, 1, 2, 10, 4, 9, 5, -1],
  [5, 2, 10, 5, 4, 2, 4, 0, 2, -1],
  [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1],
  [9, 5, 4, 2, 3, 11, -1],
  [0, 11, 2, 0, 8, 11, 4, 9, 5, -1],
  [0, 5, 4, 0, 1, 5, 2, 3, 11, -1],
  [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1],
  [10, 3, 11, 10, 1, 3, 9, 5, 4, -1],
  [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1],
  [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1],
  [5, 4, 8, 5, 8, 10, 10, 8, 11, -1],
  [9, 7, 8, 5, 7, 9, -1],
  [9, 3, 0, 9, 5, 3, 5, 7, 3, -1],
  [0, 7, 8, 0, 1, 7, 1, 5, 7, -1],
  [1, 5, 3, 3, 5, 7, -1],
  [9, 7, 8, 9, 5, 7, 10, 1, 2, -1],
  [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1],
  [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1],
  [2, 10, 5, 2, 5, 3, 3, 5, 7, -1],
  [7, 9, 5, 7, 8, 9, 3, 11, 2, -1],
  [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1],
  [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1],
  [11, 2, 1, 11, 1, 7, 7, 1, 5, -1],
  [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1],
  [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
  [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
  [11, 10, 5, 7, 11, 5, -1],
  [10, 6, 5, -1],
  [0, 8, 3, 5, 10, 6, -1],
  [9, 0, 1, 5, 10, 6, -1],
  [1, 8, 3, 1, 9, 8, 5, 10, 6, -1],
  [1, 6, 5, 2, 6, 1, -1],
  [1, 6, 5, 1, 2, 6, 3, 0, 8, -1],
  [9, 6, 5, 9, 0, 6, 0, 2, 6, -1],
  [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1],
  [2, 3, 11, 10, 6, 5, -1],
  [11, 0, 8, 11, 2, 0, 10, 6, 5, -1],
  [0, 1, 9, 2, 3, 11, 5, 10, 6, -1],
  [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1],
  [6, 3, 11, 6, 5, 3, 5, 1, 3, -1],
  [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1],
  [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1],
  [6, 5, 9, 6, 9, 11, 11, 9, 8, -1],
  [5, 10, 6, 4, 7, 8, -1],
  [4, 3, 0, 4, 7, 3, 6, 5, 10, -1],
  [1, 9, 0, 5, 10, 6, 8, 4, 7, -1],
  [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1],
  [6, 1, 2, 6, 5, 1, 4, 7, 8, -1],
  [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1],
  [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1],
  [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
  [3, 11, 2, 7, 8, 4, 10, 6, 5, -1],
  [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1],
  [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1],
  [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
  [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1],
  [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
  [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
  [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1],
  [10, 4, 9, 6, 4, 10, -1],
  [4, 10, 6, 4, 9, 10, 0, 8, 3, -1],
  [10, 0, 1, 10, 6, 0, 6, 4, 0, -1],
  [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1],
  [1, 4, 9, 1, 2, 4, 2, 6, 4, -1],
  [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1],
  [0, 2, 4, 4, 2, 6, -1],
  [8, 3, 2, 8, 2, 4, 4, 2, 6, -1],
  [10, 4, 9, 10, 6, 4, 11, 2, 3, -1],
  [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1],
  [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1],
  [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
  [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1],
  [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
  [3, 11, 6, 3, 6, 0, 0, 6, 4, -1],
  [6, 4, 8, 11, 6, 8, -1],
  [7, 10, 6, 7, 8, 10, 8, 9, 10, -1],
  [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1],
  [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1],
  [10, 6, 7, 10, 7, 1, 1, 7, 3, -1],
  [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1],
  [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
  [7, 8, 0, 7, 0, 6, 6, 0, 2, -1],
  [7, 3, 2, 6, 7, 2, -1],
  [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1],
  [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
  [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
  [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1],
  [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
  [0, 9, 1, 11, 6, 7, -1],
  [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1],
  [7, 11, 6, -1],
  [7, 6, 11, -1],
  [3, 0, 8, 11, 7, 6, -1],
  [0, 1, 9, 11, 7, 6, -1],
  [8, 1, 9, 8, 3, 1, 11, 7, 6, -1],
  [10, 1, 2, 6, 11, 7, -1],
  [1, 2, 10, 3, 0, 8, 6, 11, 7, -1],
  [2, 9, 0, 2, 10, 9, 6, 11, 7, -1],
  [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1],
  [7, 2, 3, 6, 2, 7, -1],
  [7, 0, 8, 7, 6, 0, 6, 2, 0, -1],
  [2, 7, 6, 2, 3, 7, 0, 1, 9, -1],
  [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1],
  [10, 7, 6, 10, 1, 7, 1, 3, 7, -1],
  [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1],
  [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1],
  [7, 6, 10, 7, 10, 8, 8, 10, 9, -1],
  [6, 8, 4, 11, 8, 6, -1],
  [3, 6, 11, 3, 0, 6, 0, 4, 6, -1],
  [8, 6, 11, 8, 4, 6, 9, 0, 1, -1],
  [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1],
  [6, 8, 4, 6, 11, 8, 2, 10, 1, -1],
  [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1],
  [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1],
  [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
  [8, 2, 3, 8, 4, 2, 4, 6, 2, -1],
  [0, 4, 2, 4, 6, 2, -1],
  [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1],
  [1, 9, 4, 1, 4, 2, 2, 4, 6, -1],
  [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1],
  [10, 1, 0, 10, 0, 6, 6, 0, 4, -1],
  [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
  [10, 9, 4, 6, 10, 4, -1],
  [4, 9, 5, 7, 6, 11, -1],
  [0, 8, 3, 4, 9, 5, 11, 7, 6, -1],
  [5, 0, 1, 5, 4, 0, 7, 6, 11, -1],
  [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1],
  [9, 5, 4, 10, 1, 2, 7, 6, 11, -1],
  [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1],
  [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1],
  [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
  [7, 2, 3, 7, 6, 2, 5, 4, 9, -1],
  [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1],
  [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1],
  [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
  [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1],
  [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
  [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
  [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1],
  [6, 9, 5, 6, 11, 9, 11, 8, 9, -1],
  [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1],
  [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1],
  [6, 11, 3, 6, 3, 5, 5, 3, 1, -1],
  [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1],
  [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
  [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
  [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1],
  [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1],
  [9, 5, 6, 9, 6, 0, 0, 6, 2, -1],
  [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
  [1, 5, 6, 2, 1, 6, -1],
  [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
  [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1],
  [0, 3, 8, 5, 6, 10, -1],
  [10, 5, 6, -1],
  [11, 5, 10, 7, 5, 11, -1],
  [11, 5, 10, 11, 7, 5, 8, 3, 0, -1],
  [5, 11, 7, 5, 10, 11, 1, 9, 0, -1],
  [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1],
  [11, 1, 2, 11, 7, 1, 7, 5, 1, -1],
  [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1],
  [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1],
  [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
  [2, 5, 10, 2, 3, 5, 3, 7, 5, -1],
  [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1],
  [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1],
  [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
  [1, 3, 5, 3, 7, 5, -1],
  [0, 8, 7, 0, 7, 1, 1, 7, 5, -1],
  [9, 0, 3, 9, 3, 5, 5, 3, 7, -1],
  [9, 8, 7, 5, 9, 7, -1],
  [5, 8, 4, 5, 10, 8, 10, 11, 8, -1],
  [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1],
  [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1],
  [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
  [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1],
  [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
  [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
  [9, 4, 5, 2, 11, 3, -1],
  [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1],
  [5, 10, 2, 5, 2, 4, 4, 2, 0, -1],
  [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
  [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1],
  [8, 4, 5, 8, 5, 3, 3, 5, 1, -1],
  [0, 4, 5, 1, 0, 5, -1],
  [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1],
  [9, 4, 5, -1],
  [4, 11, 7, 4, 9, 11, 9, 10, 11, -1],
  [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1],
  [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1],
  [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
  [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1],
  [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
  [11, 7, 4, 11, 4, 2, 2, 4, 0, -1],
  [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1],
  [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1],
  [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
  [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
  [1, 10, 2, 8, 7, 4, -1],
  [4, 9, 1, 4, 1, 7, 7, 1, 3, -1],
  [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1],
  [4, 0, 3, 7, 4, 3, -1],
  [4, 8, 7, -1],
  [9, 10, 8, 10, 11, 8, -1],
  [3, 0, 9, 3, 9, 11, 11, 9, 10, -1],
  [0, 1, 10, 0, 10, 8, 8, 10, 11, -1],
  [3, 1, 10, 11, 3, 10, -1],
  [1, 2, 11, 1, 11, 9, 9, 11, 8, -1],
  [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1],
  [0, 2, 11, 8, 0, 11, -1],
  [3, 2, 11, -1],
  [2, 3, 8, 2, 8, 10, 10, 8, 9, -1],
  [9, 10, 2, 0, 9, 2, -1],
  [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1],
  [1, 10, 2, -1],
  [1, 3, 8, 9, 1, 8, -1],
  [0, 9, 1, -1],
  [0, 3, 8, -1],
  [-1]
];

// ============================================================================
// High-Resolution Surface Mesh Extraction from SDF per Layer
// Uses Marching Cubes + Tangential Laplacian Relaxation + Newton-Raphson
// SDF Isosurface Projection for ultra-smooth, artifact-free organic geometry.
// ============================================================================
export function generateLayerMesh(layerCfg) {
  const { id, bbox, gridRes } = layerCfg;
  const [nx, ny, nz] = gridRes;
  const dx = (bbox.maxX - bbox.minX) / (nx - 1);
  const dy = (bbox.maxY - bbox.minY) / (ny - 1);
  const dz = (bbox.maxZ - bbox.minZ) / (nz - 1);

  // Sample SDF on grid (nudge near-zero values to avoid degenerate corner sliver triangles)
  const nxy = nx * ny;
  const field = new Float32Array(nx * ny * nz);
  for (let k = 0; k < nz; k++) {
    const z = bbox.minZ + k * dz;
    const kOff = k * nxy;
    for (let j = 0; j < ny; j++) {
      const y = bbox.minY + j * dy;
      const jOff = kOff + j * nx;
      for (let i = 0; i < nx; i++) {
        const x = bbox.minX + i * dx;
        let val = evalLayerSDF(id, x, y, z);
        if (Math.abs(val) < 1e-4) val = -1e-4;
        field[jOff + i] = val;
      }
    }
  }

  // Deduplicated vertex map along grid edges (3 edges per grid vertex: X, Y, Z)
  const edgeVertMap = new Int32Array(nx * ny * nz * 3);
  edgeVertMap.fill(-1);

  const positions = []; // [x, y, z, ...]
  const indices = [];

  function getEdgeVertex(i0, j0, k0, dir, i1, j1, k1) {
    const key = (k0 * nxy + j0 * nx + i0) * 3 + dir;
    let vIdx = edgeVertMap[key];
    if (vIdx !== -1) return vIdx;

    const idx0 = k0 * nxy + j0 * nx + i0;
    const idx1 = k1 * nxy + j1 * nx + i1;
    const v0 = field[idx0];
    const v1 = field[idx1];
    const t = clamp(v0 / (v0 - v1 + 1e-12), 0.0, 1.0);

    const x0 = bbox.minX + i0 * dx, y0 = bbox.minY + j0 * dy, z0 = bbox.minZ + k0 * dz;
    const x1 = bbox.minX + i1 * dx, y1 = bbox.minY + j1 * dy, z1 = bbox.minZ + k1 * dz;

    vIdx = positions.length / 3;
    positions.push(
      x0 + t * (x1 - x0),
      y0 + t * (y1 - y0),
      z0 + t * (z1 - z0)
    );
    edgeVertMap[key] = vIdx;
    return vIdx;
  }

  const edgeVerts = new Int32Array(12);

  for (let k = 0; k < nz - 1; k++) {
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const c0 = k * nxy + j * nx + i;
        const c1 = c0 + 1;
        const c2 = k * nxy + (j + 1) * nx + i + 1;
        const c3 = k * nxy + (j + 1) * nx + i;
        const c4 = (k + 1) * nxy + j * nx + i;
        const c5 = c4 + 1;
        const c6 = (k + 1) * nxy + (j + 1) * nx + i + 1;
        const c7 = (k + 1) * nxy + (j + 1) * nx + i;

        let cubeIndex = 0;
        if (field[c0] < 0.0) cubeIndex |= 1;
        if (field[c1] < 0.0) cubeIndex |= 2;
        if (field[c2] < 0.0) cubeIndex |= 4;
        if (field[c3] < 0.0) cubeIndex |= 8;
        if (field[c4] < 0.0) cubeIndex |= 16;
        if (field[c5] < 0.0) cubeIndex |= 32;
        if (field[c6] < 0.0) cubeIndex |= 64;
        if (field[c7] < 0.0) cubeIndex |= 128;

        const edgeMask = MC_EDGE_TABLE[cubeIndex];
        if (edgeMask === 0) continue;

        if (edgeMask & 1)    edgeVerts[0]  = getEdgeVertex(i, j, k, 0, i + 1, j, k);
        if (edgeMask & 2)    edgeVerts[1]  = getEdgeVertex(i + 1, j, k, 1, i + 1, j + 1, k);
        if (edgeMask & 4)    edgeVerts[2]  = getEdgeVertex(i, j + 1, k, 0, i + 1, j + 1, k);
        if (edgeMask & 8)    edgeVerts[3]  = getEdgeVertex(i, j, k, 1, i, j + 1, k);
        if (edgeMask & 16)   edgeVerts[4]  = getEdgeVertex(i, j, k + 1, 0, i + 1, j, k + 1);
        if (edgeMask & 32)   edgeVerts[5]  = getEdgeVertex(i + 1, j, k + 1, 1, i + 1, j + 1, k + 1);
        if (edgeMask & 64)   edgeVerts[6]  = getEdgeVertex(i, j + 1, k + 1, 0, i + 1, j + 1, k + 1);
        if (edgeMask & 128)  edgeVerts[7]  = getEdgeVertex(i, j, k + 1, 1, i, j + 1, k + 1);
        if (edgeMask & 256)  edgeVerts[8]  = getEdgeVertex(i, j, k, 2, i, j, k + 1);
        if (edgeMask & 512)  edgeVerts[9]  = getEdgeVertex(i + 1, j, k, 2, i + 1, j, k + 1);
        if (edgeMask & 1024) edgeVerts[10] = getEdgeVertex(i + 1, j + 1, k, 2, i + 1, j + 1, k + 1);
        if (edgeMask & 2048) edgeVerts[11] = getEdgeVertex(i, j + 1, k, 2, i, j + 1, k + 1);

        const triRow = MC_TRI_TABLE[cubeIndex];
        for (let t = 0; triRow[t] !== -1; t += 3) {
          // Outward-facing winding
          indices.push(edgeVerts[triRow[t]], edgeVerts[triRow[t + 2]], edgeVerts[triRow[t + 1]]);
        }
      }
    }
  }

  const numVerts = positions.length / 3;
  const posArr = new Float32Array(positions);

  // Build vertex neighbor accumulator for tangential Laplacian relaxation
  const accumX = new Float32Array(numVerts);
  const accumY = new Float32Array(numVerts);
  const accumZ = new Float32Array(numVerts);
  const counts = new Uint16Array(numVerts);

  // 4 passes of Tangential Relaxation + Exact SDF Newton-Raphson Surface Projection
  for (let pass = 0; pass < 4; pass++) {
    accumX.fill(0);
    accumY.fill(0);
    accumZ.fill(0);
    counts.fill(0);

    for (let t = 0; t < indices.length; t += 3) {
      const a = indices[t], b = indices[t + 1], c = indices[t + 2];
      const ax = posArr[a * 3], ay = posArr[a * 3 + 1], az = posArr[a * 3 + 2];
      const bx = posArr[b * 3], by = posArr[b * 3 + 1], bz = posArr[b * 3 + 2];
      const cx = posArr[c * 3], cy = posArr[c * 3 + 1], cz = posArr[c * 3 + 2];

      accumX[a] += bx + cx; accumY[a] += by + cy; accumZ[a] += bz + cz; counts[a] += 2;
      accumX[b] += ax + cx; accumY[b] += ay + cy; accumZ[b] += az + cz; counts[b] += 2;
      accumX[c] += ax + bx; accumY[c] += ay + by; accumZ[c] += az + bz; counts[c] += 2;
    }

    for (let v = 0; v < numVerts; v++) {
      const cnt = counts[v];
      if (cnt === 0) continue;
      let px = posArr[v * 3], py = posArr[v * 3 + 1], pz = posArr[v * 3 + 2];
      const avgX = accumX[v] / cnt, avgY = accumY[v] / cnt, avgZ = accumZ[v] / cnt;
      const [nx0, ny0, nz0] = evalLayerNormal(id, px, py, pz, 0.004);

      // Move tangentially + slightly along normal to polish thin rims
      const diffX = avgX - px, diffY = avgY - py, diffZ = avgZ - pz;
      const dotN = diffX * nx0 + diffY * ny0 + diffZ * nz0;
      px += (diffX - dotN * nx0 * 0.65) * 0.48;
      py += (diffY - dotN * ny0 * 0.65) * 0.48;
      pz += (diffZ - dotN * nz0 * 0.65) * 0.48;

      // Project onto exact f(p) = 0 surface via Newton-Raphson
      for (let nr = 0; nr < 2; nr++) {
        const d = evalLayerSDF(id, px, py, pz);
        const [gx, gy, gz] = evalLayerNormal(id, px, py, pz, 0.003);
        px -= d * gx * 0.85;
        py -= d * gy * 0.85;
        pz -= d * gz * 0.85;
      }

      posArr[v * 3] = px;
      posArr[v * 3 + 1] = py;
      posArr[v * 3 + 2] = pz;
    }
  }

  // Compute smoothed analytical SDF rest normals & optical thickness at each vertex
  const restNormals = new Float32Array(numVerts * 3);
  const thickness = new Float32Array(numVerts);

  for (let v = 0; v < numVerts; v++) {
    const px = posArr[v * 3], py = posArr[v * 3 + 1], pz = posArr[v * 3 + 2];
    const [nx0, ny0, nz0] = evalLayerNormal(id, px, py, pz, 0.0045);
    restNormals[v * 3] = nx0;
    restNormals[v * 3 + 1] = ny0;
    restNormals[v * 3 + 2] = nz0;
    thickness[v] = evalLayerThickness(id, px, py, pz, nx0, ny0, nz0);
  }

  return {
    layerId: id,
    numVerts,
    restPositions: posArr,
    restNormals,
    thickness,
    indices: new Uint32Array(indices)
  };
}

// ============================================================================
// Treat Template Configurations & Snuffle Ruffle Pocket Locator
// ============================================================================
export const TREAT_TEMPLATE_CONFIGS = [
  {
    id: 5,
    name: "Freeze-Dried Egg Yolk Cube",
    shortName: "Egg Yolk Cube",
    colorHex: "#F2A93B",
    massG: 2.2,
    volCm3: 2.6,
    radius: 0.068,
    bbox: { minX: -0.10, maxX: 0.10, minY: -0.09, maxY: 0.09, minZ: -0.10, maxZ: 0.10 },
    gridRes: [20, 18, 20]
  },
  {
    id: 6,
    name: "Freeze-Dried Salmon Bite",
    shortName: "Salmon Bite",
    colorHex: "#DF6D53",
    massG: 2.5,
    volCm3: 2.8,
    radius: 0.072,
    bbox: { minX: -0.10, maxX: 0.10, minY: -0.09, maxY: 0.09, minZ: -0.10, maxZ: 0.10 },
    gridRes: [20, 18, 20]
  },
  {
    id: 7,
    name: "Crunchy Star Kibble",
    shortName: "Star Kibble",
    colorHex: "#BD7438",
    massG: 2.8,
    volCm3: 2.9,
    radius: 0.075,
    bbox: { minX: -0.11, maxX: 0.11, minY: -0.08, maxY: 0.08, minZ: -0.11, maxZ: 0.11 },
    gridRes: [22, 18, 22]
  }
];

export function generateTreatTemplates() {
  return TREAT_TEMPLATE_CONFIGS.map(cfg => generateLayerMesh(cfg));
}

// Compute the 9 natural ruffle pocket cavities around Layer 1 (Matcha Snuffle Fabric)
// where the upper fold (yCenter1) and lower fold (yCenter2) flare apart widest.
export function computeSnufflePocketSlots() {
  const numPockets = 9;
  const slots = [];
  const sector = (Math.PI * 2.0) / numPockets;

  for (let k = 0; k < numPockets; k++) {
    const tStart = k * sector;
    let bestTheta = tStart + sector * 0.5;
    let bestGap = -1e9;
    let bestYMid = 0.0;
    const r = 0.875 + (k % 2) * 0.020; // ~0.875 - 0.895 so treats peek out of the green ruffle folds

    const flare = smoothstep(0.22, 0.90, r);
    for (let s = 0; s < 36; s++) {
      const th = tStart + (s / 36) * sector;
      const wave1 = flare * (
        0.068 * Math.sin(9.0 * th + 1.5 * r) +
        0.022 * Math.cos(18.0 * th - 2.0 * r)
      );
      const wave2 = flare * (
        0.070 * Math.cos(9.0 * th - 1.4 * r + 0.8) +
        0.022 * Math.sin(18.0 * th + 1.8 * r)
      );
      const y1 = 0.048 + wave1;
      const y2 = -0.048 + wave2;
      const gap = y1 - y2;
      if (gap > bestGap) {
        bestGap = gap;
        bestTheta = th;
        bestYMid = 0.5 * (y1 + y2);
      }
    }

    const templateIdx = k % 3; // Alternate Egg Yolk Cube (0), Salmon Bite (1), Star Kibble (2)
    const cfg = TREAT_TEMPLATE_CONFIGS[templateIdx];

    slots.push({
      pocketIdx: k,
      templateIdx,
      layerId: cfg.id,
      theta: bestTheta,
      radius: r,
      localPos: [
        r * Math.cos(bestTheta),
        bestYMid,
        r * Math.sin(bestTheta)
      ],
      // Natural tilted rest angles inside the ruffle fold
      restEuler: [
        0.28 * Math.sin(k * 2.1 + 0.4),
        bestTheta + 0.5 * Math.cos(k * 1.7),
        0.24 * Math.cos(k * 2.7 - 0.3)
      ],
      massG: cfg.massG,
      volCm3: cfg.volCm3,
      boundRadius: cfg.radius
    });
  }

  return slots;
}

