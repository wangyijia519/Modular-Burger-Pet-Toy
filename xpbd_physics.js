// ============================================================================
// xpbd_physics.js
// Real-time CPU XPBD Corotational Shape-Matching & Tetrahedral Volume Solver
// 10 Substeps @ 60 Hz with Distinct Multi-Material Zones & Trilinear Skinning
// ============================================================================

import { LAYER_CONFIGS, MM_PER_UNIT, TREAT_TEMPLATE_CONFIGS, generateTreatTemplates, computeSnufflePocketSlots } from './sdf_mesh.js';

// Distinct XPBD material zone profiles for each of the 5 coaxial layers
export const MATERIAL_ZONES = [
  // Layer 0: Top Shell — Firm semi-rigid silicone rubber (Shore 65A)
  {
    name: "Firm Semi-Rigid Rubber",
    edgeCompliance: 1.4e-5,
    volCompliance: 2.0e-6,
    shapeStiffness: 0.76,
    anchorStiffness: 0.14,
    localDamping: 0.18
  },
  // Layer 1: Matcha Snuffle Fabric — Soft low-bending-stiffness felt cloth
  {
    name: "Low-Bending Felt Fabric",
    edgeCompliance: 3.2e-4,
    volCompliance: 4.5e-4,
    shapeStiffness: 0.055, // Very low bending/shape stiffness so ruffles deform locally!
    anchorStiffness: 0.048,
    localDamping: 0.24
  },
  // Layer 2: Coral-Red TPE Chew Ring — Highly elastic jelly-like TPE (Shore 22A)
  {
    name: "Jelly-Like Hyper-Elastic TPE",
    edgeCompliance: 8.5e-4, // Stretches and squishes dramatically when grabbed
    volCompliance: 1.2e-6,  // Strong Poisson volume preservation -> bulges & thins
    shapeStiffness: 0.135,  // Recoils with lively elastic oscillation
    anchorStiffness: 0.052,
    localDamping: 0.06      // Low internal damping for lively jelly recoil
  },
  // Layer 3: Warm-Orange Frisbee Disc — Medium-flexibility elastomer (Shore 45A)
  {
    name: "Medium-Flex Elastomer",
    edgeCompliance: 1.1e-4,
    volCompliance: 5.0e-6,
    shapeStiffness: 0.44,
    anchorStiffness: 0.11,
    localDamping: 0.14
  },
  // Layer 4: Bottom Shell — Firm semi-rigid silicone rubber (Shore 65A)
  {
    name: "Firm Semi-Rigid Rubber",
    edgeCompliance: 1.4e-5,
    volCompliance: 2.0e-6,
    shapeStiffness: 0.76,
    anchorStiffness: 0.15,
    localDamping: 0.18
  }
];

// Conformal alternating-parity 5-tetrahedra decompositions inside a unit hexahedron (corners 0..7)
// Corner bitmask: 0:(0,0,0), 1:(1,0,0), 2:(0,1,0), 3:(1,1,0), 4:(0,0,1), 5:(1,0,1), 6:(0,1,1), 7:(1,1,1)
const HEX_TET_PARITY = [
  [
    [0, 1, 2, 4],
    [3, 2, 1, 7],
    [5, 4, 7, 1],
    [6, 7, 4, 2],
    [1, 2, 4, 7]
  ],
  [
    [1, 3, 0, 5],
    [2, 0, 3, 6],
    [4, 6, 5, 0],
    [7, 5, 6, 3],
    [0, 3, 5, 6]
  ]
];

// 22 edges per hexahedral cell (12 box edges + 6 face diagonals + 4 body diagonals)
const HEX_EDGE_PAIRS = [
  // 12 structural edges
  [0,1],[2,3],[4,5],[6,7],
  [0,2],[1,3],[4,6],[5,7],
  [0,4],[1,5],[2,6],[3,7],
  // 6 shear face diagonals
  [0,3],[4,7],[0,5],[2,7],[0,6],[1,7],
  // 4 body diagonals
  [0,7],[1,6],[2,5],[3,4]
];

// Robust Quaternion Polar Decomposition (Matthias Müller 2016)
// Extracts exact rotation matrix R (3x3) from deformation/moment matrix A (3x3)
function extractRotationPolar(A, qOut, ROut) {
  let qx = qOut[0], qy = qOut[1], qz = qOut[2], qw = qOut[3];

  for (let iter = 0; iter < 5; iter++) {
    // Convert quaternion q to rotation matrix R
    const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
    const xx = qx * x2, xy = qx * y2, xz = qx * z2;
    const yy = qy * y2, yz = qy * z2, zz = qz * z2;
    const wx = qw * x2, wy = qw * y2, wz = qw * z2;

    // Columns of R
    const r00 = 1.0 - (yy + zz), r10 = xy + wz,         r20 = xz - wy;
    const r01 = xy - wz,         r11 = 1.0 - (xx + zz), r21 = yz + wx;
    const r02 = xz + wy,         r12 = yz - wx,         r22 = 1.0 - (xx + yy);

    // Columns of A
    const a00 = A[0], a10 = A[3], a20 = A[6];
    const a01 = A[1], a11 = A[4], a21 = A[7];
    const a02 = A[2], a12 = A[5], a22 = A[8];

    // omega = (R_col0 x A_col0 + R_col1 x A_col1 + R_col2 x A_col2) / (1 + |R_col0.A_col0 + R_col1.A_col1 + R_col2.A_col2|)
    const ox = (r10 * a20 - r20 * a10) + (r11 * a21 - r21 * a11) + (r12 * a22 - r22 * a12);
    const oy = (r20 * a00 - r00 * a20) + (r21 * a01 - r01 * a21) + (r22 * a02 - r02 * a22);
    const oz = (r00 * a10 - r10 * a00) + (r01 * a11 - r11 * a01) + (r02 * a12 - r12 * a02);

    const dotSum = (r00 * a00 + r10 * a10 + r20 * a20) +
                   (r01 * a01 + r11 * a11 + r21 * a21) +
                   (r02 * a02 + r12 * a12 + r22 * a22);

    const denom = Math.abs(dotSum) + 1e-6;
    const wxStep = ox / denom;
    const wyStep = oy / denom;
    const wzStep = oz / denom;
    const wLen2 = wxStep * wxStep + wyStep * wyStep + wzStep * wzStep;
    if (wLen2 < 1e-9) break;

    const wLen = Math.sqrt(wLen2);
    const halfAngle = 0.5 * wLen;
    const s = Math.sin(halfAngle) / wLen;
    const dqX = wxStep * s, dqY = wyStep * s, dqZ = wzStep * s, dqW = Math.cos(halfAngle);

    // q = dq * q
    const nqx = dqW * qx + dqX * qw + dqY * qz - dqZ * qy;
    const nqy = dqW * qy - dqX * qz + dqY * qw + dqZ * qx;
    const nqz = dqW * qz + dqX * qy - dqY * qx + dqZ * qw;
    const nqw = dqW * qw - dqX * qx - dqY * qy - dqZ * qz;

    const invN = 1.0 / (Math.sqrt(nqx * nqx + nqy * nqy + nqz * nqz + nqw * nqw) + 1e-12);
    qx = nqx * invN; qy = nqy * invN; qz = nqz * invN; qw = nqw * invN;
  }

  qOut[0] = qx; qOut[1] = qy; qOut[2] = qz; qOut[3] = qw;

  const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
  const xx = qx * x2, xy = qx * y2, xz = qx * z2;
  const yy = qy * y2, yz = qy * z2, zz = qz * z2;
  const wx = qw * x2, wy = qw * y2, wz = qw * z2;

  ROut[0] = 1.0 - (yy + zz); ROut[1] = xy - wz;         ROut[2] = xz + wy;
  ROut[3] = xy + wz;         ROut[4] = 1.0 - (xx + zz); ROut[5] = yz - wx;
  ROut[6] = xz - wy;         ROut[7] = yz + wx;         ROut[8] = 1.0 - (xx + yy);
}

export class BurgerXPBDSolver {
  constructor(layerMeshes, treatTemplates = null) {
    this.layerMeshes = layerMeshes;
    this.treatTemplates = treatTemplates || generateTreatTemplates();
    this.pocketSlots = computeSnufflePocketSlots();
    this.clickFeltToHide = false;

    this.mode = 'exploded'; // 'exploded' | 'burger' | 'sphere'
    this.explosionGapMm = 26.0; // Default exploded gap in mm
    this.stiffnessScale = 1.0;
    this.dampingScale = 0.35;
    this.slowMo = false;
    this.paused = false;

    this.substeps = 10;
    this.dtFrame = 1.0 / 60.0;

    // Grab interaction state (supports both soft-body lattice grab and individual treat grab)
    this.grabState = {
      active: false,
      treatIdx: -1,
      layerId: -1,
      nodeIndices: [],
      nodeWeights: [],
      localOffsets: [],
      targetPos: [0, 0, 0],
      currentCenter: [0, 0, 0],
      forceN: 0.0
    };

    this._buildLatticeAndSkinning();
    this.resetSimulation(true);
  }

  // Compute vertical world Y offset for a given layer in the current mode
  getLayerTargetY(layerId) {
    const cfg = LAYER_CONFIGS[layerId];
    if (this.mode === 'sphere') {
      return cfg.ySphere;
    }
    const gapUnits = (this.mode === 'exploded')
      ? (this.explosionGapMm / MM_PER_UNIT)
      : (this.explosionGapMm / MM_PER_UNIT);
    return cfg.yBurger + cfg.explodeRank * gapUnits;
  }

  isLayerActive(layerId) {
    if (this.mode === 'sphere') {
      return layerId === 0 || layerId === 4;
    }
    return true;
  }

  _buildLatticeAndSkinning() {
    const nodesLocalRest = []; // [x, y, z] in layer local frame
    const nodesLayer = [];     // layerId per node
    const nodesMass = [];      // mass in grams

    const hexCells = [];       // { layerId, corners: Int32Array(8), restCenter: [x,y,z], quat: [0,0,0,1] }
    const edgeMap = new Map();
    const edges = [];          // { i, j, restLen, compliance, layerId }
    const tets = [];           // { i0, i1, i2, i3, restVol, compliance, layerId }

    // Per-vertex skinning data across all 5 layer meshes + 9 snuffle treats
    let layerRenderVerts = 0;
    let totalRenderVerts = 0;
    let totalRenderIndices = 0;
    for (const m of this.layerMeshes) {
      layerRenderVerts += m.numVerts;
      totalRenderVerts += m.numVerts;
      totalRenderIndices += m.indices.length;
    }
    for (const slot of this.pocketSlots) {
      const tm = this.treatTemplates[slot.templateIdx];
      totalRenderVerts += tm.numVerts;
      totalRenderIndices += tm.indices.length;
    }
    this.layerRenderVerts = layerRenderVerts;
    this.totalRenderVerts = totalRenderVerts;
    this.totalRenderIndices = totalRenderIndices;

    // Interleaved GPU vertex buffer: 12 floats (48 bytes) per vertex
    // [posX, posY, posZ, layerId, normX, normY, normZ, contactAO, restX, restY, restZ, thickness]
    this.skinnedVertexData = new Float32Array(totalRenderVerts * 12);
    this.combinedIndices = new Uint32Array(totalRenderIndices);

    this.skinCellIdx = new Int32Array(layerRenderVerts);
    this.skinWeights = new Float32Array(layerRenderVerts * 8);
    this.layerDrawRanges = []; // { layerId, firstIndex, indexCount, baseVertex, vertCount }
    this.treatDrawRanges = []; // { treatIdx, layerId, firstIndex, indexCount, baseVertex, vertCount }

    let globalVertOffset = 0;
    let globalIndexOffset = 0;

    for (let lid = 0; lid < 5; lid++) {
      const cfg = LAYER_CONFIGS[lid];
      const mesh = this.layerMeshes[lid];
      const { bbox, hexRes } = cfg;
      const [hx, hy, hz] = hexRes;

      const pad = 0.025;
      const minX = bbox.minX - pad, maxX = bbox.maxX + pad;
      const minY = bbox.minY - pad, maxY = bbox.maxY + pad;
      const minZ = bbox.minZ - pad, maxZ = bbox.maxZ + pad;
      const dx = (maxX - minX) / hx;
      const dy = (maxY - minY) / hy;
      const dz = (maxZ - minZ) / hz;

      // Mark which hex cells contain or neighbor mesh vertices
      const cellOccupied = new Uint8Array(hx * hy * hz);
      const vPos = mesh.restPositions;

      for (let v = 0; v < mesh.numVerts; v++) {
        const vx = vPos[v * 3], vy = vPos[v * 3 + 1], vz = vPos[v * 3 + 2];
        const ci = Math.max(0, Math.min(hx - 1, Math.floor((vx - minX) / dx)));
        const cj = Math.max(0, Math.min(hy - 1, Math.floor((vy - minY) / dy)));
        const ck = Math.max(0, Math.min(hz - 1, Math.floor((vz - minZ) / dz)));
        cellOccupied[(ck * hy + cj) * hx + ci] = 1;
      }

      // Create lattice nodes and active hex cells for this layer
      const gridNodeIdx = new Int32Array((hx + 1) * (hy + 1) * (hz + 1));
      gridNodeIdx.fill(-1);
      const cellToGlobalHex = new Int32Array(hx * hy * hz);
      cellToGlobalHex.fill(-1);

      const getOrMakeNode = (gi, gj, gk) => {
        const gKey = (gk * (hy + 1) + gj) * (hx + 1) + gi;
        let nIdx = gridNodeIdx[gKey];
        if (nIdx !== -1) return nIdx;
        nIdx = nodesLayer.length;
        const nx = minX + gi * dx;
        const ny = minY + gj * dy;
        const nz = minZ + gk * dz;
        nodesLocalRest.push(nx, ny, nz);
        nodesLayer.push(lid);
        nodesMass.push(0.0);
        gridNodeIdx[gKey] = nIdx;
        return nIdx;
      };

      const activeLayerCells = [];

      for (let ck = 0; ck < hz; ck++) {
        for (let cj = 0; cj < hy; cj++) {
          for (let ci = 0; ci < hx; ci++) {
            const cKey = (ck * hy + cj) * hx + ci;
            if (!cellOccupied[cKey]) continue;

            const corners = new Int32Array([
              getOrMakeNode(ci,     cj,     ck),
              getOrMakeNode(ci + 1, cj,     ck),
              getOrMakeNode(ci,     cj + 1, ck),
              getOrMakeNode(ci + 1, cj + 1, ck),
              getOrMakeNode(ci,     cj,     ck + 1),
              getOrMakeNode(ci + 1, cj,     ck + 1),
              getOrMakeNode(ci,     cj + 1, ck + 1),
              getOrMakeNode(ci + 1, cj + 1, ck + 1)
            ]);

            let cx = 0, cy = 0, cz = 0;
            for (let c = 0; c < 8; c++) {
              const ni = corners[c] * 3;
              cx += nodesLocalRest[ni];
              cy += nodesLocalRest[ni + 1];
              cz += nodesLocalRest[ni + 2];
            }
            cx *= 0.125; cy *= 0.125; cz *= 0.125;

            const globalHexIdx = hexCells.length;
            cellToGlobalHex[cKey] = globalHexIdx;
            activeLayerCells.push({ ci, cj, ck, globalHexIdx });

            hexCells.push({
              layerId: lid,
              corners,
              restCenter: [cx, cy, cz],
              quat: new Float32Array([0, 0, 0, 1]),
              rotMat: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
              normalMat: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1])
            });

            // Add 22 hex edges
            const mat = MATERIAL_ZONES[lid];
            for (let e = 0; e < HEX_EDGE_PAIRS.length; e++) {
              const [ca, cb] = HEX_EDGE_PAIRS[e];
              const na = corners[ca], nb = corners[cb];
              const minN = Math.min(na, nb), maxN = Math.max(na, nb);
              const eKey = `${minN}_${maxN}`;
              if (!edgeMap.has(eKey)) {
                const ax = nodesLocalRest[na * 3], ay = nodesLocalRest[na * 3 + 1], az = nodesLocalRest[na * 3 + 2];
                const bx = nodesLocalRest[nb * 3], by = nodesLocalRest[nb * 3 + 1], bz = nodesLocalRest[nb * 3 + 2];
                const len = Math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2);
                edgeMap.set(eKey, edges.length);
                edges.push({
                  i: minN,
                  j: maxN,
                  restLen: len,
                  compliance: mat.edgeCompliance,
                  layerId: lid
                });
              }
            }

            // Add 5 tetrahedra for volume conservation (alternating parity for symmetry)
            const tetTable = HEX_TET_PARITY[(ci + cj + ck) & 1];
            const cm3PerUnit3 = (MM_PER_UNIT * 0.1) ** 3; // (5.5 cm)^3 = 166.375 cm^3 per unit^3
            const fillFactors = [0.195, 0.145, 0.245, 0.245, 0.195];
            const fillFactor = fillFactors[lid];
            let hexVolSum = 0.0;

            for (let t = 0; t < 5; t++) {
              const ti = tetTable[t];
              const i0 = corners[ti[0]], i1 = corners[ti[1]], i2 = corners[ti[2]], i3 = corners[ti[3]];
              const x0 = nodesLocalRest[i0 * 3], y0 = nodesLocalRest[i0 * 3 + 1], z0 = nodesLocalRest[i0 * 3 + 2];
              const x1 = nodesLocalRest[i1 * 3] - x0, y1 = nodesLocalRest[i1 * 3 + 1] - y0, z1 = nodesLocalRest[i1 * 3 + 2] - z0;
              const x2 = nodesLocalRest[i2 * 3] - x0, y2 = nodesLocalRest[i2 * 3 + 1] - y0, z2 = nodesLocalRest[i2 * 3 + 2] - z0;
              const x3 = nodesLocalRest[i3 * 3] - x0, y3 = nodesLocalRest[i3 * 3 + 1] - y0, z3 = nodesLocalRest[i3 * 3 + 2] - z0;

              const crossX = y1 * z2 - z1 * y2;
              const crossY = z1 * x2 - x1 * z2;
              const crossZ = x1 * y2 - y1 * x2;
              const vol = (crossX * x3 + crossY * y3 + crossZ * z3) / 6.0;
              hexVolSum += Math.abs(vol);

              tets.push({
                i0: vol >= 0 ? i0 : i0,
                i1: vol >= 0 ? i1 : i2,
                i2: vol >= 0 ? i2 : i1,
                i3,
                restVol: Math.abs(vol),
                fillFactor,
                compliance: mat.volCompliance,
                layerId: lid
              });
            }

            // Distribute hexahedral cell mass equally across all 8 corners (eliminates 5-tet corner mass bias)
            const cornerMassG = (hexVolSum * cm3PerUnit3 * cfg.densityGPerCm3 * fillFactor) * 0.125;
            for (let c = 0; c < 8; c++) {
              nodesMass[corners[c]] += cornerMassG;
            }
          }
        }
      }

      // Bind every mesh vertex of this layer to its containing active hex cell
      for (let v = 0; v < mesh.numVerts; v++) {
        const vx = vPos[v * 3], vy = vPos[v * 3 + 1], vz = vPos[v * 3 + 2];
        let ci = Math.max(0, Math.min(hx - 1, Math.floor((vx - minX) / dx)));
        let cj = Math.max(0, Math.min(hy - 1, Math.floor((vy - minY) / dy)));
        let ck = Math.max(0, Math.min(hz - 1, Math.floor((vz - minZ) / dz)));
        let hexIdx = cellToGlobalHex[(ck * hy + cj) * hx + ci];

        if (hexIdx === -1) {
          // Fallback: nearest active cell in layer
          let bestDist = 1e9;
          for (let a = 0; a < activeLayerCells.length; a++) {
            const ac = activeLayerCells[a];
            const dist = (ac.ci - ci) ** 2 + (ac.cj - cj) ** 2 + (ac.ck - ck) ** 2;
            if (dist < bestDist) {
              bestDist = dist;
              ci = ac.ci; cj = ac.cj; ck = ac.ck;
              hexIdx = ac.globalHexIdx;
            }
          }
        }

        const u = Math.max(-0.15, Math.min(1.15, (vx - (minX + ci * dx)) / dx));
        const vCoord = Math.max(-0.15, Math.min(1.15, (vy - (minY + cj * dy)) / dy));
        const w = Math.max(-0.15, Math.min(1.15, (vz - (minZ + ck * dz)) / dz));

        const u0 = 1.0 - u, u1 = u;
        const v0 = 1.0 - vCoord, v1 = vCoord;
        const w0 = 1.0 - w, w1 = w;

        const gv = globalVertOffset + v;
        this.skinCellIdx[gv] = hexIdx;
        const wOff = gv * 8;
        this.skinWeights[wOff + 0] = u0 * v0 * w0;
        this.skinWeights[wOff + 1] = u1 * v0 * w0;
        this.skinWeights[wOff + 2] = u0 * v1 * w0;
        this.skinWeights[wOff + 3] = u1 * v1 * w0;
        this.skinWeights[wOff + 4] = u0 * v0 * w1;
        this.skinWeights[wOff + 5] = u1 * v0 * w1;
        this.skinWeights[wOff + 6] = u0 * v1 * w1;
        this.skinWeights[wOff + 7] = u1 * v1 * w1;

        // Pre-fill static vertex attributes (layerId, restPos, thickness)
        const vBase = gv * 12;
        this.skinnedVertexData[vBase + 3]  = lid;
        this.skinnedVertexData[vBase + 8]  = vx;
        this.skinnedVertexData[vBase + 9]  = vy;
        this.skinnedVertexData[vBase + 10] = vz;
        this.skinnedVertexData[vBase + 11] = mesh.thickness[v];
      }

      // Copy triangle indices with global vertex offset
      const mIdx = mesh.indices;
      for (let i = 0; i < mIdx.length; i++) {
        this.combinedIndices[globalIndexOffset + i] = mIdx[i] + globalVertOffset;
      }

      this.layerDrawRanges.push({
        layerId: lid,
        firstIndex: globalIndexOffset,
        indexCount: mIdx.length,
        baseVertex: globalVertOffset,
        vertCount: mesh.numVerts
      });

      globalVertOffset += mesh.numVerts;
      globalIndexOffset += mIdx.length;
    }

    // Store typed arrays for fast XPBD loop
    this.numNodes = nodesLayer.length;
    this.nodeLayer = new Int32Array(nodesLayer);
    this.nodeLocalRest = new Float32Array(nodesLocalRest);
    this.nodeRestPos = new Float32Array(this.numNodes * 3);
    this.nodePos = new Float32Array(this.numNodes * 3);
    this.nodePrevPos = new Float32Array(this.numNodes * 3);
    this.nodeVel = new Float32Array(this.numNodes * 3);
    this.nodeMass = new Float32Array(nodesMass);
    this.nodeInvMass = new Float32Array(this.numNodes);
    this.nodeAnchorWeight = new Float32Array(this.numNodes);

    for (let i = 0; i < this.numNodes; i++) {
      const m = Math.max(0.05, this.nodeMass[i]);
      this.nodeMass[i] = m;
      this.nodeInvMass[i] = 1.0 / m;

      // Compute radial distance in rest space for posture tether profile
      const rx = this.nodeLocalRest[i * 3];
      const rz = this.nodeLocalRest[i * 3 + 2];
      const r = Math.sqrt(rx * rx + rz * rz);
      const lid = this.nodeLayer[i];
      if (lid === 1) {
        // Matcha Snuffle Felt: inner core is anchored coaxially, outer ruffles are super free & floppy!
        const innerFactor = Math.max(0.06, 1.0 - Math.min(1.0, Math.max(0.0, (r - 0.30) / 0.65)) * 0.90);
        this.nodeAnchorWeight[i] = innerFactor;
      } else if (lid === 2) {
        // Coral-Red TPE Chew Ring: outer bumps are extra stretchy & bouncy
        this.nodeAnchorWeight[i] = (r > 0.78) ? 0.45 : 0.85;
      } else {
        this.nodeAnchorWeight[i] = 1.0;
      }
    }

    this.hexCells = hexCells;
    this.edges = edges;
    this.tets = tets;

    // Populate the 9 Snuffle Treat instances & bind them to Layer 1 ruffle pockets
    this._buildSnuffleTreats(globalVertOffset, globalIndexOffset);

    // Build inter-layer coaxial coupling pairs for Burger Snuffle Mode & Rolling Sphere Mode
    this._buildInterLayerCouplings();

    // Build lattice wireframe line indices for "Show Mesh" overlay
    const lineIndices = [];
    for (let e = 0; e < edges.length; e++) {
      // Only include primary structural box edges (shorter than diagonal)
      if (edges[e].restLen < 0.36) {
        lineIndices.push(edges[e].i, edges[e].j);
      }
    }
    this.latticeLineIndices = new Uint32Array(lineIndices);
  }

  _buildSnuffleTreats(startVertOffset, startIndexOffset) {
    let globalVertOffset = startVertOffset;
    let globalIndexOffset = startIndexOffset;
    this.treats = [];

    // Collect Layer 1 hex cells for pocket binding
    const layer1Cells = [];
    for (let c = 0; c < this.hexCells.length; c++) {
      if (this.hexCells[c].layerId === 1) {
        layer1Cells.push(c);
      }
    }

    const cfg1 = LAYER_CONFIGS[1];
    const pad = 0.025;
    const minX = cfg1.bbox.minX - pad, maxX = cfg1.bbox.maxX + pad;
    const minY = cfg1.bbox.minY - pad, maxY = cfg1.bbox.maxY + pad;
    const minZ = cfg1.bbox.minZ - pad, maxZ = cfg1.bbox.maxZ + pad;
    const dx = (maxX - minX) / cfg1.hexRes[0];
    const dy = (maxY - minY) / cfg1.hexRes[1];
    const dz = (maxZ - minZ) / cfg1.hexRes[2];

    for (let k = 0; k < this.pocketSlots.length; k++) {
      const slot = this.pocketSlots[k];
      const tm = this.treatTemplates[slot.templateIdx];
      const [px, py, pz] = slot.localPos;

      // Find nearest active hex cell in Layer 1
      let bestHexIdx = layer1Cells[0];
      let bestD2 = 1e9;
      for (const hcIdx of layer1Cells) {
        const rc = this.hexCells[hcIdx].restCenter;
        const d2 = (rc[0] - px) ** 2 + (rc[1] - py) ** 2 + (rc[2] - pz) ** 2;
        if (d2 < bestD2) {
          bestD2 = d2;
          bestHexIdx = hcIdx;
        }
      }

      const hostCell = this.hexCells[bestHexIdx];
      const c0 = hostCell.corners[0] * 3;
      const cellMinX = this.nodeLocalRest[c0];
      const cellMinY = this.nodeLocalRest[c0 + 1];
      const cellMinZ = this.nodeLocalRest[c0 + 2];

      const u = Math.max(-0.2, Math.min(1.2, (px - cellMinX) / dx));
      const v = Math.max(-0.2, Math.min(1.2, (py - cellMinY) / dy));
      const w = Math.max(-0.2, Math.min(1.2, (pz - cellMinZ) / dz));
      const u0 = 1.0 - u, u1 = u;
      const v0 = 1.0 - v, v1 = v;
      const w0 = 1.0 - w, w1 = w;
      const pocketWeights = new Float32Array([
        u0 * v0 * w0, u1 * v0 * w0, u0 * v1 * w0, u1 * v1 * w0,
        u0 * v0 * w1, u1 * v0 * w1, u0 * v1 * w1, u1 * v1 * w1
      ]);

      // Find nearby upper & lower Layer 1 felt nodes around this pocket
      const upperNodes = [];
      const lowerNodes = [];
      for (let i = 0; i < this.numNodes; i++) {
        if (this.nodeLayer[i] !== 1) continue;
        const nx = this.nodeLocalRest[i * 3];
        const ny = this.nodeLocalRest[i * 3 + 1];
        const nz = this.nodeLocalRest[i * 3 + 2];
        const dHoriz2 = (nx - px) ** 2 + (nz - pz) ** 2;
        if (dHoriz2 < 0.28 * 0.28) {
          const wNode = Math.max(0.1, 1.0 - Math.sqrt(dHoriz2) / 0.28);
          if (ny >= py) {
            upperNodes.push({ nodeIdx: i, weight: wNode });
          } else {
            lowerNodes.push({ nodeIdx: i, weight: wNode });
          }
        }
      }

      // Pre-fill static vertex attributes for this treat in skinnedVertexData
      for (let vIdx = 0; vIdx < tm.numVerts; vIdx++) {
        const gv = globalVertOffset + vIdx;
        const vBase = gv * 12;
        this.skinnedVertexData[vBase + 3]  = slot.layerId; // 5, 6, or 7
        this.skinnedVertexData[vBase + 8]  = tm.restPositions[vIdx * 3];
        this.skinnedVertexData[vBase + 9]  = tm.restPositions[vIdx * 3 + 1];
        this.skinnedVertexData[vBase + 10] = tm.restPositions[vIdx * 3 + 2];
        this.skinnedVertexData[vBase + 11] = tm.thickness[vIdx];
      }

      for (let i = 0; i < tm.indices.length; i++) {
        this.combinedIndices[globalIndexOffset + i] = tm.indices[i] + globalVertOffset;
      }

      const drawRange = {
        treatIdx: k,
        layerId: slot.layerId,
        firstIndex: globalIndexOffset,
        indexCount: tm.indices.length,
        baseVertex: globalVertOffset,
        vertCount: tm.numVerts
      };
      this.treatDrawRanges.push(drawRange);

      this.treats.push({
        idx: k,
        layerId: slot.layerId,
        template: tm,
        drawRange,
        pocketLocalPos: slot.localPos.slice(),
        theta: slot.theta,
        restEuler: slot.restEuler.slice(),
        massG: slot.massG,
        volCm3: slot.volCm3,
        boundRadius: slot.boundRadius,
        hexIdx: bestHexIdx,
        pocketWeights,
        upperNodes,
        lowerNodes,
        state: 'hidden', // 'hidden' (tucked in green felt) | 'free' (ejected/floor) | 'inactive'
        pos: [slot.localPos[0], slot.localPos[1] + this.getLayerTargetY(1), slot.localPos[2]],
        vel: [0.0, 0.0, 0.0],
        euler: slot.restEuler.slice(),
        angVel: [0.0, 0.0, 0.0]
      });

      globalVertOffset += tm.numVerts;
      globalIndexOffset += tm.indices.length;
    }
  }

  _buildInterLayerCouplings() {
    // Pairs of nodes between adjacent layers in Burger mode, and between Layer 0 & Layer 4 rims in Sphere mode
    this.burgerLinks = [];
    this.sphereLinks = [];

    const layerNodes = [[], [], [], [], []];
    for (let i = 0; i < this.numNodes; i++) {
      layerNodes[this.nodeLayer[i]].push(i);
    }

    // Helper to link closest nodes between layer A and layer B in Burger mode
    const linkLayers = (la, lb, maxRadialDist, maxVertDist, stiffness) => {
      const ya = LAYER_CONFIGS[la].yBurger;
      const yb = LAYER_CONFIGS[lb].yBurger;
      for (const na of layerNodes[la]) {
        const ax = this.nodeLocalRest[na * 3];
        const ay = this.nodeLocalRest[na * 3 + 1] + ya;
        const az = this.nodeLocalRest[na * 3 + 2];
        let bestB = -1;
        let bestD2 = 1e9;
        for (const nb of layerNodes[lb]) {
          const bx = this.nodeLocalRest[nb * 3];
          const by = this.nodeLocalRest[nb * 3 + 1] + yb;
          const bz = this.nodeLocalRest[nb * 3 + 2];
          if (Math.abs(ay - by) > maxVertDist) continue;
          const dr2 = (ax - bx) ** 2 + (az - bz) ** 2;
          if (dr2 < maxRadialDist * maxRadialDist && dr2 + (ay - by) ** 2 < bestD2) {
            bestD2 = dr2 + (ay - by) ** 2;
            bestB = nb;
          }
        }
        if (bestB !== -1) {
          this.burgerLinks.push({
            a: na,
            b: bestB,
            restDx: this.nodeLocalRest[bestB * 3] - ax,
            restDy: (this.nodeLocalRest[bestB * 3 + 1] + yb) - ay,
            restDz: this.nodeLocalRest[bestB * 3 + 2] - az,
            stiffness
          });
        }
      }
    };

    linkLayers(0, 1, 0.28, 0.26, 0.18);
    linkLayers(1, 2, 0.28, 0.28, 0.16);
    linkLayers(2, 3, 0.28, 0.28, 0.20);
    linkLayers(3, 4, 0.28, 0.28, 0.24);
    linkLayers(2, 4, 0.25, 0.26, 0.22);

    // Sphere mode links: lock Top Shell (Layer 0) lower rim to Bottom Shell (Layer 4) upper rim
    const y0s = LAYER_CONFIGS[0].ySphere;
    const y4s = LAYER_CONFIGS[4].ySphere;
    for (const n0 of layerNodes[0]) {
      const y0 = this.nodeLocalRest[n0 * 3 + 1];
      if (y0 > 0.22) continue; // Only rim nodes
      const x0 = this.nodeLocalRest[n0 * 3];
      const z0 = this.nodeLocalRest[n0 * 3 + 2];
      let best4 = -1, bestD2 = 1e9;
      for (const n4 of layerNodes[4]) {
        const y4 = this.nodeLocalRest[n4 * 3 + 1];
        if (y4 < -0.22) continue;
        const dx = this.nodeLocalRest[n4 * 3] - x0;
        const dz = this.nodeLocalRest[n4 * 3 + 2] - z0;
        const d2 = dx * dx + dz * dz;
        if (d2 < 0.08 && d2 < bestD2) {
          bestD2 = d2;
          best4 = n4;
        }
      }
      if (best4 !== -1) {
        this.sphereLinks.push({
          a: n0,
          b: best4,
          restDx: this.nodeLocalRest[best4 * 3] - x0,
          restDy: (this.nodeLocalRest[best4 * 3 + 1] + y4s) - (y0 + y0s),
          restDz: this.nodeLocalRest[best4 * 3 + 2] - z0,
          stiffness: 0.45
        });
      }
    }
  }

  // Update target rest positions when mode or explosionGap changes
  updateRestTargets() {
    for (let i = 0; i < this.numNodes; i++) {
      const lid = this.nodeLayer[i];
      const yOff = this.getLayerTargetY(lid);
      this.nodeRestPos[i * 3]     = this.nodeLocalRest[i * 3];
      this.nodeRestPos[i * 3 + 1] = this.nodeLocalRest[i * 3 + 1] + yOff;
      this.nodeRestPos[i * 3 + 2] = this.nodeLocalRest[i * 3 + 2];
    }
  }

  isTreatVisible(treatIdx) {
    const tr = this.treats[treatIdx];
    if (!tr || tr.state === 'inactive') return false;
    if (this.mode === 'sphere' && tr.state === 'hidden') return false;
    return true;
  }

  getPocketWorldPos(treatIdx) {
    const tr = this.treats[treatIdx];
    const cell = this.hexCells[tr.hexIdx];
    const cn = cell.corners;
    const w = tr.pocketWeights;
    let px = 0, py = 0, pz = 0;
    for (let k = 0; k < 8; k++) {
      const ni = cn[k] * 3;
      const wk = w[k];
      px += wk * this.nodePos[ni];
      py += wk * this.nodePos[ni + 1];
      pz += wk * this.nodePos[ni + 2];
    }
    return [px, py, pz];
  }

  stuffSingleTreat(treatIdx, withImpulse = true) {
    const tr = this.treats[treatIdx];
    if (!tr) return;
    tr.state = 'hidden';
    tr.vel[0] = 0; tr.vel[1] = 0; tr.vel[2] = 0;
    tr.angVel[0] = 0; tr.angVel[1] = 0; tr.angVel[2] = 0;
    tr.euler[0] = tr.restEuler[0];
    tr.euler[1] = tr.restEuler[1];
    tr.euler[2] = tr.restEuler[2];
    const pWorld = this.getPocketWorldPos(treatIdx);
    tr.pos[0] = pWorld[0]; tr.pos[1] = pWorld[1]; tr.pos[2] = pWorld[2];

    if (withImpulse && this.isLayerActive(1)) {
      for (const u of tr.upperNodes) {
        this.nodeVel[u.nodeIdx * 3 + 1] += 1.65 * u.weight;
      }
      for (const l of tr.lowerNodes) {
        this.nodeVel[l.nodeIdx * 3 + 1] -= 1.65 * l.weight;
      }
    }
  }

  popSingleTreat(treatIdx) {
    const tr = this.treats[treatIdx];
    if (!tr || tr.state !== 'hidden') return;
    const pWorld = this.getPocketWorldPos(treatIdx);
    tr.pos[0] = pWorld[0];
    tr.pos[1] = pWorld[1];
    tr.pos[2] = pWorld[2];
    tr.state = 'free';

    const r = Math.hypot(pWorld[0], pWorld[2]) + 1e-6;
    const ux = pWorld[0] / r, uz = pWorld[2] / r;
    const tangX = -uz, tangZ = ux;
    const jitter = ((treatIdx % 3) - 1) * 0.45;
    const outSpeed = 2.05 + (treatIdx % 4) * 0.25;

    tr.vel[0] = ux * outSpeed + tangX * jitter;
    tr.vel[1] = 1.25 + (treatIdx % 2) * 0.35;
    tr.vel[2] = uz * outSpeed + tangZ * jitter;

    tr.angVel[0] = 5.5 * Math.sin(treatIdx * 2.3 + 0.5);
    tr.angVel[1] = 4.2 * Math.cos(treatIdx * 1.9);
    tr.angVel[2] = -5.0 * Math.cos(treatIdx * 3.1 - 0.4);

    // Local felt ruffle recoil when treat pops out
    for (const u of tr.upperNodes) {
      this.nodeVel[u.nodeIdx * 3 + 1] += 2.2 * u.weight;
      this.nodeVel[u.nodeIdx * 3]     += ux * 1.2 * u.weight;
      this.nodeVel[u.nodeIdx * 3 + 2] += uz * 1.2 * u.weight;
    }
    for (const l of tr.lowerNodes) {
      this.nodeVel[l.nodeIdx * 3 + 1] -= 1.8 * l.weight;
    }
  }

  stuffAllTreats(withImpulse = true) {
    for (let k = 0; k < this.treats.length; k++) {
      this.stuffSingleTreat(k, withImpulse);
    }
    this.updateSkinnedMesh();
  }

  snuffleDislodgeTreats(count = 3) {
    // Collect currently hidden treats
    const hiddenIndices = [];
    for (let k = 0; k < this.treats.length; k++) {
      if (this.treats[k].state === 'hidden') {
        hiddenIndices.push(k);
      }
    }
    // If none are hidden, stuff all 9 first so the user always sees treats pop out!
    if (hiddenIndices.length === 0) {
      this.stuffAllTreats(true);
      for (let k = 0; k < this.treats.length; k++) hiddenIndices.push(k);
    }

    // Prioritize front-facing pockets (highest Z) so the user clearly sees them pop out toward the camera
    hiddenIndices.sort((a, b) => this.treats[b].pocketLocalPos[2] - this.treats[a].pocketLocalPos[2]);
    const numToPop = Math.min(count, hiddenIndices.length);
    for (let i = 0; i < numToPop; i++) {
      this.popSingleTreat(hiddenIndices[i]);
    }
  }

  clearAllTreats() {
    for (let k = 0; k < this.treats.length; k++) {
      this.treats[k].state = 'inactive';
      this.treats[k].vel.fill(0);
      this.treats[k].angVel.fill(0);
    }
    this.updateSkinnedMesh();
  }

  resetSimulation(snapImmediate = true) {
    this.updateRestTargets();
    for (let i = 0; i < this.numNodes * 3; i++) {
      this.nodePos[i] = this.nodeRestPos[i];
      this.nodePrevPos[i] = this.nodeRestPos[i];
      this.nodeVel[i] = 0.0;
    }
    for (let c = 0; c < this.hexCells.length; c++) {
      this.hexCells[c].quat.set([0, 0, 0, 1]);
    }
    if (this.treats) {
      for (let k = 0; k < this.treats.length; k++) {
        if (this.treats[k].state !== 'inactive') {
          this.stuffSingleTreat(k, false);
        }
      }
    }
    if (snapImmediate) {
      this.updateSkinnedMesh();
    }
  }

  // Apply interactive Poke impulse (radial squeeze + vertical bounce + rotational twist on elastic layers)
  triggerPoke() {
    for (let i = 0; i < this.numNodes; i++) {
      const lid = this.nodeLayer[i];
      if (!this.isLayerActive(lid)) continue;

      const x = this.nodePos[i * 3];
      const y = this.nodePos[i * 3 + 1];
      const z = this.nodePos[i * 3 + 2];
      const r = Math.sqrt(x * x + z * z) + 1e-6;
      const ux = x / r, uz = z / r;

      if (lid === 2) {
        // Coral-Red TPE Chew Ring: dramatic outward/asymmetric jelly squish & bounce!
        const asym = Math.cos(3.0 * Math.atan2(z, x)) * 1.6 + 0.9;
        this.nodeVel[i * 3]     += ux * 3.8 * asym;
        this.nodeVel[i * 3 + 1] += (Math.sin(4.0 * Math.atan2(z, x)) * 2.2) + 1.4;
        this.nodeVel[i * 3 + 2] += uz * 3.8 * asym;
      } else if (lid === 1) {
        // Matcha Snuffle Felt: flutter & ruffle wave impulse!
        const wave = Math.sin(7.0 * Math.atan2(z, x) + 1.2);
        const rimFactor = Math.max(0.0, (r - 0.32) / 0.70);
        this.nodeVel[i * 3]     += ux * 2.6 * rimFactor * wave;
        this.nodeVel[i * 3 + 1] += 3.5 * rimFactor * Math.cos(5.0 * Math.atan2(z, x));
        this.nodeVel[i * 3 + 2] += uz * 2.6 * rimFactor * wave;
      } else if (lid === 3) {
        // Warm-Orange Frisbee: saddle-mode flex impulse
        const saddle = Math.cos(2.0 * Math.atan2(z, x));
        this.nodeVel[i * 3]     += ux * 1.8 * saddle;
        this.nodeVel[i * 3 + 1] += 2.6 * saddle * r;
        this.nodeVel[i * 3 + 2] += uz * 1.8 * saddle;
      } else {
        // Top & Bottom Shells: firm impact compression & rock
        const dirY = (lid === 0) ? -2.6 : 2.2;
        this.nodeVel[i * 3]     += ux * 1.4 + (lid === 0 ? 0.8 : -0.5);
        this.nodeVel[i * 3 + 1] += dirY + x * 1.5;
        this.nodeVel[i * 3 + 2] += uz * 1.4;
      }
    }
  }

  // Raycast against treats & active lattice nodes to begin interactive 3D grabbing or treat stuffing
  beginGrab(rayOrigin, rayDir, isShiftClick = false) {
    // 1. If "Click Felt to Hide" mode or Shift-click is active, check if ray hits near any Layer 1 pocket
    if ((this.clickFeltToHide || isShiftClick) && this.isLayerActive(1)) {
      let bestPocket = -1;
      let bestPocketD2 = 0.28 * 0.28;
      let bestPocketT = 1e9;
      for (let k = 0; k < this.treats.length; k++) {
        const pWorld = this.getPocketWorldPos(k);
        const px = pWorld[0] - rayOrigin[0];
        const py = pWorld[1] - rayOrigin[1];
        const pz = pWorld[2] - rayOrigin[2];
        const t = px * rayDir[0] + py * rayDir[1] + pz * rayDir[2];
        if (t < 0.2) continue;
        const cx = px - t * rayDir[0];
        const cy = py - t * rayDir[1];
        const cz = pz - t * rayDir[2];
        const d2 = cx * cx + cy * cy + cz * cz;
        if (d2 < bestPocketD2 && t < bestPocketT + 0.25) {
          bestPocketD2 = d2;
          bestPocketT = t;
          bestPocket = k;
        }
      }
      if (bestPocket !== -1) {
        if (this.treats[bestPocket].state === 'hidden') {
          this.popSingleTreat(bestPocket);
        } else {
          this.stuffSingleTreat(bestPocket, true);
        }
        return true;
      }
    }

    // 2. Check if ray hits a 'free' treat on the floor/air so the user can pick it up and re-hide it in the felt!
    let bestTreat = -1;
    let bestTreatT = 1e9;
    for (let k = 0; k < this.treats.length; k++) {
      if (!this.isTreatVisible(k) || this.treats[k].state !== 'free') continue;
      const tr = this.treats[k];
      const px = tr.pos[0] - rayOrigin[0];
      const py = tr.pos[1] - rayOrigin[1];
      const pz = tr.pos[2] - rayOrigin[2];
      const t = px * rayDir[0] + py * rayDir[1] + pz * rayDir[2];
      if (t < 0.2) continue;
      const cx = px - t * rayDir[0];
      const cy = py - t * rayDir[1];
      const cz = pz - t * rayDir[2];
      const d2 = cx * cx + cy * cy + cz * cz;
      const pickR = tr.boundRadius * 2.1;
      if (d2 < pickR * pickR && t < bestTreatT) {
        bestTreatT = t;
        bestTreat = k;
      }
    }

    if (bestTreat !== -1) {
      const tr = this.treats[bestTreat];
      this.grabState.active = true;
      this.grabState.treatIdx = bestTreat;
      this.grabState.layerId = tr.layerId;
      this.grabState.nodeIndices = [];
      this.grabState.nodeWeights = [];
      this.grabState.localOffsets = [];
      this.grabState.targetPos = tr.pos.slice();
      this.grabState.currentCenter = tr.pos.slice();
      this.grabState.rayDistance = bestTreatT;
      this.grabState.forceN = 2.5;
      return true;
    }

    // 3. Raycast against soft-body lattice nodes
    let bestNode = -1;
    let bestT = 1e9;
    let bestDistSq = 0.24 * 0.24; // Grab pick radius

    for (let i = 0; i < this.numNodes; i++) {
      const lid = this.nodeLayer[i];
      if (!this.isLayerActive(lid)) continue;

      const px = this.nodePos[i * 3] - rayOrigin[0];
      const py = this.nodePos[i * 3 + 1] - rayOrigin[1];
      const pz = this.nodePos[i * 3 + 2] - rayOrigin[2];
      const t = px * rayDir[0] + py * rayDir[1] + pz * rayDir[2];
      if (t < 0.2) continue;

      const cx = px - t * rayDir[0];
      const cy = py - t * rayDir[1];
      const cz = pz - t * rayDir[2];
      const d2 = cx * cx + cy * cy + cz * cz;

      // Score combines ray proximity and front-most depth
      if (d2 < bestDistSq && (t < bestT + 0.18)) {
        if (t < bestT || d2 < bestDistSq * 0.4) {
          bestT = t;
          bestDistSq = Math.max(d2, 0.015);
          bestNode = i;
        }
      }
    }

    if (bestNode === -1) return false;

    const hitLid = this.nodeLayer[bestNode];
    const hx = this.nodePos[bestNode * 3];
    const hy = this.nodePos[bestNode * 3 + 1];
    const hz = this.nodePos[bestNode * 3 + 2];

    // Influence radius depends on material: smaller local radius for felt ruffles so pulling deforms ruffles locally!
    const grabRadius = (hitLid === 1) ? 0.30 : (hitLid === 2 ? 0.42 : 0.48);
    const r2Max = grabRadius * grabRadius;

    const nodeIndices = [];
    const nodeWeights = [];
    const localOffsets = [];

    for (let i = 0; i < this.numNodes; i++) {
      if (this.nodeLayer[i] !== hitLid) continue;
      const dx = this.nodePos[i * 3] - hx;
      const dy = this.nodePos[i * 3 + 1] - hy;
      const dz = this.nodePos[i * 3 + 2] - hz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 <= r2Max) {
        const w = (1.0 - Math.sqrt(d2) / grabRadius) ** 1.5;
        nodeIndices.push(i);
        nodeWeights.push(w);
        localOffsets.push([dx, dy, dz]);
      }
    }

    this.grabState.active = true;
    this.grabState.treatIdx = -1;
    this.grabState.layerId = hitLid;
    this.grabState.nodeIndices = nodeIndices;
    this.grabState.nodeWeights = nodeWeights;
    this.grabState.localOffsets = localOffsets;
    this.grabState.targetPos = [hx, hy, hz];
    this.grabState.currentCenter = [hx, hy, hz];
    this.grabState.rayDistance = bestT;
    this.grabState.forceN = 0.0;
    return true;
  }

  moveGrab(rayOrigin, rayDir) {
    if (!this.grabState.active) return;
    const t = this.grabState.rayDistance;
    this.grabState.targetPos[0] = rayOrigin[0] + rayDir[0] * t;
    this.grabState.targetPos[1] = Math.max(-0.98, rayOrigin[1] + rayDir[1] * t);
    this.grabState.targetPos[2] = rayOrigin[2] + rayDir[2] * t;

    // If user is dragging a 'free' treat and brings it close to an empty pocket on Layer 1, snap-tuck it in!
    if (this.grabState.treatIdx !== -1 && this.isLayerActive(1)) {
      const kGrab = this.grabState.treatIdx;
      const tr = this.treats[kGrab];
      tr.pos[0] = this.grabState.targetPos[0];
      tr.pos[1] = this.grabState.targetPos[1];
      tr.pos[2] = this.grabState.targetPos[2];
      tr.vel[0] = 0; tr.vel[1] = 0; tr.vel[2] = 0;
      this.grabState.currentCenter[0] = tr.pos[0];
      this.grabState.currentCenter[1] = tr.pos[1];
      this.grabState.currentCenter[2] = tr.pos[2];

      const pOwn = this.getPocketWorldPos(kGrab);
      const dOwn = Math.hypot(tr.pos[0] - pOwn[0], tr.pos[1] - pOwn[1], tr.pos[2] - pOwn[2]);
      if (dOwn < 0.24) {
        this.stuffSingleTreat(kGrab, true);
        this.endGrab();
      }
    }
  }

  endGrab() {
    this.grabState.active = false;
    this.grabState.treatIdx = -1;
    this.grabState.forceN = 0.0;
  }

  // Main XPBD step: 10 substeps at 60 Hz
  step() {
    this.updateRestTargets();
    if (this.paused) {
      return this.computeTelemetry();
    }

    const timeScale = this.slowMo ? 0.25 : 1.0;
    const dt = (this.dtFrame * timeScale) / this.substeps;
    const invDt = 1.0 / dt;
    const dt2 = dt * dt;

    const stiffMult = Math.max(0.15, this.stiffnessScale);
    const dampMult = Math.max(0.0, Math.min(1.0, this.dampingScale));

    const A = new Float32Array(9);
    const R = new Float32Array(9);

    for (let sub = 0; sub < this.substeps; sub++) {
      // 1. Semi-implicit Euler prediction + Restoring Posture Tether + Grab Spring
      for (let i = 0; i < this.numNodes; i++) {
        const lid = this.nodeLayer[i];
        if (!this.isLayerActive(lid)) continue;

        const idx = i * 3;
        const mat = MATERIAL_ZONES[lid];

        // Velocity damping (combines material internal damping + user damping slider)
        const velDecay = Math.max(0.82, 1.0 - (mat.localDamping * 0.4 + dampMult * 0.85) * dt * 9.0);
        let vx = this.nodeVel[idx] * velDecay;
        let vy = this.nodeVel[idx + 1] * velDecay;
        let vz = this.nodeVel[idx + 2] * velDecay;

        // Soft coaxial posture tether toward mode rest target
        const rx = this.nodeRestPos[idx];
        const ry = this.nodeRestPos[idx + 1];
        const rz = this.nodeRestPos[idx + 2];
        const kAnchor = mat.anchorStiffness * this.nodeAnchorWeight[i] * stiffMult * 420.0;

        vx += (rx - this.nodePos[idx]) * kAnchor * dt;
        vy += (ry - this.nodePos[idx + 1]) * kAnchor * dt;
        vz += (rz - this.nodePos[idx + 2]) * kAnchor * dt;

        this.nodePrevPos[idx]     = this.nodePos[idx];
        this.nodePrevPos[idx + 1] = this.nodePos[idx + 1];
        this.nodePrevPos[idx + 2] = this.nodePos[idx + 2];

        this.nodePos[idx]     += vx * dt;
        this.nodePos[idx + 1] += vy * dt;
        this.nodePos[idx + 2] += vz * dt;
      }

      // Apply user interactive grab constraint
      if (this.grabState.active && this.grabState.treatIdx === -1) {
        const { nodeIndices, nodeWeights, localOffsets, targetPos } = this.grabState;
        let sumDx = 0, sumDy = 0, sumDz = 0;
        let cx = 0, cy = 0, cz = 0, wSum = 0;
        for (let k = 0; k < nodeIndices.length; k++) {
          const ni = nodeIndices[k] * 3;
          const w = nodeWeights[k];
          const off = localOffsets[k];
          const gx = targetPos[0] + off[0];
          const gy = targetPos[1] + off[1];
          const gz = targetPos[2] + off[2];
          const pullStrength = 0.48 * w;
          sumDx += (gx - this.nodePos[ni]) * w;
          sumDy += (gy - this.nodePos[ni + 1]) * w;
          sumDz += (gz - this.nodePos[ni + 2]) * w;
          this.nodePos[ni]     += (gx - this.nodePos[ni]) * pullStrength;
          this.nodePos[ni + 1] += (gy - this.nodePos[ni + 1]) * pullStrength;
          this.nodePos[ni + 2] += (gz - this.nodePos[ni + 2]) * pullStrength;
          cx += this.nodePos[ni] * w;
          cy += this.nodePos[ni + 1] * w;
          cz += this.nodePos[ni + 2] * w;
          wSum += w;
        }
        if (wSum > 0) {
          this.grabState.currentCenter[0] = cx / wSum;
          this.grabState.currentCenter[1] = cy / wSum;
          this.grabState.currentCenter[2] = cz / wSum;
          this.grabState.forceN = Math.min(45.0, Math.sqrt(sumDx * sumDx + sumDy * sumDy + sumDz * sumDz) * 8.5);
        }
      }

      // 2. XPBD Edge Stretch / Shear Distance Constraints
      for (let e = 0; e < this.edges.length; e++) {
        const edge = this.edges[e];
        if (!this.isLayerActive(edge.layerId)) continue;

        const i3 = edge.i * 3, j3 = edge.j * 3;
        const dx = this.nodePos[j3] - this.nodePos[i3];
        const dy = this.nodePos[j3 + 1] - this.nodePos[i3 + 1];
        const dz = this.nodePos[j3 + 2] - this.nodePos[i3 + 2];
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9;

        const C = len - edge.restLen;
        const w1 = this.nodeInvMass[edge.i];
        const w2 = this.nodeInvMass[edge.j];
        const alphaTilde = (edge.compliance / stiffMult) / dt2;
        const lambda = -C / (w1 + w2 + alphaTilde);
        const corr = lambda / len;

        this.nodePos[i3]     -= w1 * corr * dx;
        this.nodePos[i3 + 1] -= w1 * corr * dy;
        this.nodePos[i3 + 2] -= w1 * corr * dz;
        this.nodePos[j3]     += w2 * corr * dx;
        this.nodePos[j3 + 1] += w2 * corr * dy;
        this.nodePos[j3 + 2] += w2 * corr * dz;
      }

      // 3. XPBD Tetrahedral Volume Constraints (C_vol = V - V_0)
      for (let t = 0; t < this.tets.length; t++) {
        const tet = this.tets[t];
        if (!this.isLayerActive(tet.layerId)) continue;

        const i0 = tet.i0 * 3, i1 = tet.i1 * 3, i2 = tet.i2 * 3, i3 = tet.i3 * 3;
        const x0 = this.nodePos[i0], y0 = this.nodePos[i0 + 1], z0 = this.nodePos[i0 + 2];
        const x10 = this.nodePos[i1] - x0, y10 = this.nodePos[i1 + 1] - y0, z10 = this.nodePos[i1 + 2] - z0;
        const x20 = this.nodePos[i2] - x0, y20 = this.nodePos[i2 + 1] - y0, z20 = this.nodePos[i2 + 2] - z0;
        const x30 = this.nodePos[i3] - x0, y30 = this.nodePos[i3 + 1] - y0, z30 = this.nodePos[i3 + 2] - z0;

        // Gradients grad1 = (x20 x x30)/6, grad2 = (x30 x x10)/6, grad3 = (x10 x x20)/6
        const g1x = (y20 * z30 - z20 * y30) * (1.0 / 6.0);
        const g1y = (z20 * x30 - x20 * z30) * (1.0 / 6.0);
        const g1z = (x20 * y30 - y20 * x30) * (1.0 / 6.0);

        const g2x = (y30 * z10 - z30 * y10) * (1.0 / 6.0);
        const g2y = (z30 * x10 - x30 * z10) * (1.0 / 6.0);
        const g2z = (x30 * y10 - y30 * x10) * (1.0 / 6.0);

        const g3x = (y10 * z20 - z10 * y20) * (1.0 / 6.0);
        const g3y = (z10 * x20 - x10 * z20) * (1.0 / 6.0);
        const g3z = (x10 * y20 - y10 * x20) * (1.0 / 6.0);

        const g0x = -(g1x + g2x + g3x);
        const g0y = -(g1y + g2y + g3y);
        const g0z = -(g1z + g2z + g3z);

        const vol = x10 * g1x + y10 * g1y + z10 * g1z;
        const C = vol - tet.restVol;

        const w0 = this.nodeInvMass[tet.i0];
        const w1 = this.nodeInvMass[tet.i1];
        const w2 = this.nodeInvMass[tet.i2];
        const w3 = this.nodeInvMass[tet.i3];

        const wGrad2 = w0 * (g0x * g0x + g0y * g0y + g0z * g0z) +
                       w1 * (g1x * g1x + g1y * g1y + g1z * g1z) +
                       w2 * (g2x * g2x + g2y * g2y + g2z * g2z) +
                       w3 * (g3x * g3x + g3y * g3y + g3z * g3z);

        if (wGrad2 > 1e-11) {
          const alphaTilde = (tet.compliance / stiffMult) / dt2;
          const lambda = -C / (wGrad2 + alphaTilde);

          this.nodePos[i0]     += w0 * lambda * g0x;
          this.nodePos[i0 + 1] += w0 * lambda * g0y;
          this.nodePos[i0 + 2] += w0 * lambda * g0z;

          this.nodePos[i1]     += w1 * lambda * g1x;
          this.nodePos[i1 + 1] += w1 * lambda * g1y;
          this.nodePos[i1 + 2] += w1 * lambda * g1z;

          this.nodePos[i2]     += w2 * lambda * g2x;
          this.nodePos[i2 + 1] += w2 * lambda * g2y;
          this.nodePos[i2 + 2] += w2 * lambda * g2z;

          this.nodePos[i3]     += w3 * lambda * g3x;
          this.nodePos[i3 + 1] += w3 * lambda * g3y;
          this.nodePos[i3 + 2] += w3 * lambda * g3z;
        }
      }

      // 4. Corotational Shape-Matching per Hexahedral Lattice Element (Polar Decomposition)
      for (let c = 0; c < this.hexCells.length; c++) {
        const cell = this.hexCells[c];
        const lid = cell.layerId;
        if (!this.isLayerActive(lid)) continue;

        const corners = cell.corners;
        const mat = MATERIAL_ZONES[lid];
        const kShape = Math.min(0.92, mat.shapeStiffness * Math.sqrt(stiffMult));

        // Compute current center of mass of the 8 hex corners
        let cx = 0, cy = 0, cz = 0;
        for (let k = 0; k < 8; k++) {
          const ni = corners[k] * 3;
          cx += this.nodePos[ni];
          cy += this.nodePos[ni + 1];
          cz += this.nodePos[ni + 2];
        }
        cx *= 0.125; cy *= 0.125; cz *= 0.125;

        const rcx = cell.restCenter[0];
        const rcy = cell.restCenter[1];
        const rcz = cell.restCenter[2];

        // Build covariance/moment matrix A_pq = sum (x_i - c)(x_i^0 - c_0)^T
        A.fill(0.0);
        for (let k = 0; k < 8; k++) {
          const ni = corners[k] * 3;
          const px = this.nodePos[ni] - cx;
          const py = this.nodePos[ni + 1] - cy;
          const pz = this.nodePos[ni + 2] - cz;

          const qx = this.nodeLocalRest[ni] - rcx;
          const qy = this.nodeLocalRest[ni + 1] - rcy;
          const qz = this.nodeLocalRest[ni + 2] - rcz;

          A[0] += px * qx; A[1] += px * qy; A[2] += px * qz;
          A[3] += py * qx; A[4] += py * qy; A[5] += py * qz;
          A[6] += pz * qx; A[7] += pz * qy; A[8] += pz * qz;
        }

        extractRotationPolar(A, cell.quat, R);
        cell.rotMat.set(R);

        // Blend rotational shape-matching goal with upright coaxial orientation
        const uprightBlend = (lid === 1) ? 0.08 : (lid === 2 ? 0.18 : 0.35);
        for (let k = 0; k < 8; k++) {
          const ni = corners[k] * 3;
          const qx = this.nodeLocalRest[ni] - rcx;
          const qy = this.nodeLocalRest[ni + 1] - rcy;
          const qz = this.nodeLocalRest[ni + 2] - rcz;

          const rx = R[0] * qx + R[1] * qy + R[2] * qz;
          const ry = R[3] * qx + R[4] * qy + R[5] * qz;
          const rz = R[6] * qx + R[7] * qy + R[8] * qz;

          const gx = cx + rx * (1.0 - uprightBlend) + qx * uprightBlend;
          const gy = cy + ry * (1.0 - uprightBlend) + qy * uprightBlend;
          const gz = cz + rz * (1.0 - uprightBlend) + qz * uprightBlend;

          this.nodePos[ni]     += (gx - this.nodePos[ni]) * kShape * 0.25;
          this.nodePos[ni + 1] += (gy - this.nodePos[ni + 1]) * kShape * 0.25;
          this.nodePos[ni + 2] += (gz - this.nodePos[ni + 2]) * kShape * 0.25;
        }
      }

      // 5. Inter-Layer Coupling & Floor Collision
      if (this.mode === 'burger' && this.explosionGapMm < 4.0) {
        const blend = 1.0 - this.explosionGapMm / 4.0;
        for (let l = 0; l < this.burgerLinks.length; l++) {
          const link = this.burgerLinks[l];
          const a3 = link.a * 3, b3 = link.b * 3;
          const ex = (this.nodePos[b3] - this.nodePos[a3]) - link.restDx;
          const ey = (this.nodePos[b3 + 1] - this.nodePos[a3 + 1]) - link.restDy;
          const ez = (this.nodePos[b3 + 2] - this.nodePos[a3 + 2]) - link.restDz;
          const k = link.stiffness * blend * 0.35;
          this.nodePos[a3]     += ex * k;
          this.nodePos[a3 + 1] += ey * k;
          this.nodePos[a3 + 2] += ez * k;
          this.nodePos[b3]     -= ex * k;
          this.nodePos[b3 + 1] -= ey * k;
          this.nodePos[b3 + 2] -= ez * k;
        }
      } else if (this.mode === 'sphere') {
        for (let l = 0; l < this.sphereLinks.length; l++) {
          const link = this.sphereLinks[l];
          const a3 = link.a * 3, b3 = link.b * 3;
          const ex = (this.nodePos[b3] - this.nodePos[a3]) - link.restDx;
          const ey = (this.nodePos[b3 + 1] - this.nodePos[a3 + 1]) - link.restDy;
          const ez = (this.nodePos[b3 + 2] - this.nodePos[a3 + 2]) - link.restDz;
          const k = link.stiffness * 0.45;
          this.nodePos[a3]     += ex * k;
          this.nodePos[a3 + 1] += ey * k;
          this.nodePos[a3 + 2] += ez * k;
          this.nodePos[b3]     -= ex * k;
          this.nodePos[b3 + 1] -= ey * k;
          this.nodePos[b3 + 2] -= ez * k;
        }
      }

      // Floor plane contact below Bottom Shell (Layer 4) base
      const floorY = Math.min(-1.05, this.getLayerTargetY(4) - 0.92);
      for (let i = 0; i < this.numNodes; i++) {
        if (!this.isLayerActive(this.nodeLayer[i])) continue;
        const idx = i * 3;
        if (this.nodePos[idx + 1] < floorY) {
          this.nodePos[idx + 1] = floorY;
          // Tangential ground friction
          this.nodePos[idx]     = this.nodePrevPos[idx] * 0.65 + this.nodePos[idx] * 0.35;
          this.nodePos[idx + 2] = this.nodePrevPos[idx + 2] * 0.65 + this.nodePos[idx + 2] * 0.35;
        }
      }

      // 6. XPBD Velocity Update: v = (x - x_prev) / dt
      for (let i = 0; i < this.numNodes; i++) {
        if (!this.isLayerActive(this.nodeLayer[i])) continue;
        const idx = i * 3;
        this.nodeVel[idx]     = (this.nodePos[idx]     - this.nodePrevPos[idx])     * invDt;
        this.nodeVel[idx + 1] = (this.nodePos[idx + 1] - this.nodePrevPos[idx + 1]) * invDt;
        this.nodeVel[idx + 2] = (this.nodePos[idx + 2] - this.nodePrevPos[idx + 2]) * invDt;
      }
    }

    // 7. Step Snuffle Treats (Pocket Retention, Foraging Pop-Out & Studio Floor Bounce)
    this._stepTreats(this.dtFrame * timeScale);

    // Update skinned surface mesh vertices after the 10 substeps
    this.updateSkinnedMesh();
    return this.computeTelemetry();
  }

  _stepTreats(dtTreat) {
    if (!this.treats) return;
    // Match visual studio floor height in webgpu_renderer.js: min(-1.036, yLayer4 - 0.855)
    const studioFloorY = Math.min(-1.036, this.getLayerTargetY(4) - 0.855);

    for (let k = 0; k < this.treats.length; k++) {
      const tr = this.treats[k];
      if (!this.isTreatVisible(k)) continue;

      if (tr.state === 'hidden') {
        // Track host ruffle pocket on Layer 1
        const pWorld = this.getPocketWorldPos(k);
        tr.pos[0] = pWorld[0];
        tr.pos[1] = pWorld[1];
        tr.pos[2] = pWorld[2];

        // Check if user is actively pulling the green snuffle fabric (Layer 1) near this pocket
        if (this.grabState.active && this.grabState.treatIdx === -1 && this.grabState.layerId === 1) {
          const gc = this.grabState.currentCenter;
          const gt = this.grabState.targetPos;
          const distToGrab = Math.hypot(pWorld[0] - gc[0], pWorld[1] - gc[1], pWorld[2] - gc[2]);
          const pullDist = Math.hypot(gt[0] - gc[0], gt[1] - gc[1], gt[2] - gc[2]);
          if (distToGrab < 0.36 && pullDist > 0.085) {
            this.popSingleTreat(k);
            continue;
          }
        }

        // Also check local felt ruffle deformation around pocket (if pulled open > 11 mm, dislodge treat!)
        let maxDisp = 0.0;
        for (const u of tr.upperNodes) {
          const ni = u.nodeIdx * 3;
          const dx = this.nodePos[ni] - this.nodeRestPos[ni];
          const dy = this.nodePos[ni + 1] - this.nodeRestPos[ni + 1];
          const dz = this.nodePos[ni + 2] - this.nodeRestPos[ni + 2];
          const d = Math.hypot(dx, dy, dz);
          if (d > maxDisp) maxDisp = d;
        }
        if (maxDisp > 0.19) {
          this.popSingleTreat(k);
        }
      } else if (tr.state === 'free') {
        // If actively grabbed by the user, position is controlled by moveGrab
        if (this.grabState.active && this.grabState.treatIdx === k) {
          tr.pos[0] = this.grabState.targetPos[0];
          tr.pos[1] = this.grabState.targetPos[1];
          tr.pos[2] = this.grabState.targetPos[2];
          tr.angVel[0] *= 0.9; tr.angVel[1] *= 0.9; tr.angVel[2] *= 0.9;
          continue;
        }

        // Integrate gravity & air drag
        tr.vel[1] -= 9.6 * dtTreat;
        tr.vel[0] *= 0.992;
        tr.vel[1] *= 0.992;
        tr.vel[2] *= 0.992;

        tr.pos[0] += tr.vel[0] * dtTreat;
        tr.pos[1] += tr.vel[1] * dtTreat;
        tr.pos[2] += tr.vel[2] * dtTreat;

        tr.euler[0] += tr.angVel[0] * dtTreat;
        tr.euler[1] += tr.angVel[1] * dtTreat;
        tr.euler[2] += tr.angVel[2] * dtTreat;

        // Radial deflection off the burger's lower bowl / frisbee cylinder so treats roll around the burger
        const rHoriz = Math.hypot(tr.pos[0], tr.pos[2]) + 1e-6;
        const yBotShell = this.getLayerTargetY(4);
        if (tr.pos[1] < yBotShell + 0.08 && tr.pos[1] > yBotShell - 0.78 && rHoriz < 1.04) {
          const ux = tr.pos[0] / rHoriz;
          const uz = tr.pos[2] / rHoriz;
          tr.pos[0] = ux * 1.04;
          tr.pos[2] = uz * 1.04;
          const vRad = tr.vel[0] * ux + tr.vel[2] * uz;
          if (vRad < 0) {
            tr.vel[0] -= 1.35 * vRad * ux;
            tr.vel[2] -= 1.35 * vRad * uz;
          }
        }

        // Studio floor collision & rolling friction
        const minTreatY = studioFloorY + tr.boundRadius * 0.62;
        if (tr.pos[1] < minTreatY) {
          tr.pos[1] = minTreatY;
          if (tr.vel[1] < -0.35) {
            tr.vel[1] = -tr.vel[1] * 0.42;
            tr.angVel[0] += tr.vel[2] * 2.5;
            tr.angVel[2] -= tr.vel[0] * 2.5;
          } else {
            tr.vel[1] = 0.0;
          }
          // Ground friction
          tr.vel[0] *= 0.84;
          tr.vel[2] *= 0.84;
          tr.angVel[0] *= 0.82;
          tr.angVel[1] *= 0.82;
          tr.angVel[2] *= 0.82;
          if (Math.hypot(tr.vel[0], tr.vel[2]) < 0.02) {
            tr.vel[0] = 0; tr.vel[2] = 0;
            tr.angVel[0] = 0; tr.angVel[1] = 0; tr.angVel[2] = 0;
          }
        }
      }
    }
  }

  // Skin the high-resolution SDF surface mesh over the deformed hexahedral lattice
  updateSkinnedMesh() {
    if (!this.nodeNormalMat || this.nodeNormalMat.length !== this.numNodes * 9) {
      this.nodeNormalMat = new Float32Array(this.numNodes * 9);
    }
    this.nodeNormalMat.fill(0.0);

    // 1. Compute Jacobian cofactor normal matrix for each active hex cell & accumulate to corner nodes
    for (let c = 0; c < this.hexCells.length; c++) {
      const cell = this.hexCells[c];
      if (!this.isLayerActive(cell.layerId)) continue;

      const cn = cell.corners;
      const c0 = cn[0] * 3, c1 = cn[1] * 3, c2 = cn[2] * 3, c3 = cn[3] * 3;
      const c4 = cn[4] * 3, c5 = cn[5] * 3, c6 = cn[6] * 3, c7 = cn[7] * 3;

      // Partial derivatives dX/du, dX/dv, dX/dw across the hex cell
      const ux = (this.nodePos[c1] - this.nodePos[c0]) + (this.nodePos[c3] - this.nodePos[c2]) +
                 (this.nodePos[c5] - this.nodePos[c4]) + (this.nodePos[c7] - this.nodePos[c6]);
      const uy = (this.nodePos[c1 + 1] - this.nodePos[c0 + 1]) + (this.nodePos[c3 + 1] - this.nodePos[c2 + 1]) +
                 (this.nodePos[c5 + 1] - this.nodePos[c4 + 1]) + (this.nodePos[c7 + 1] - this.nodePos[c6 + 1]);
      const uz = (this.nodePos[c1 + 2] - this.nodePos[c0 + 2]) + (this.nodePos[c3 + 2] - this.nodePos[c2 + 2]) +
                 (this.nodePos[c5 + 2] - this.nodePos[c4 + 2]) + (this.nodePos[c7 + 2] - this.nodePos[c6 + 2]);

      const vx = (this.nodePos[c2] - this.nodePos[c0]) + (this.nodePos[c3] - this.nodePos[c1]) +
                 (this.nodePos[c6] - this.nodePos[c4]) + (this.nodePos[c7] - this.nodePos[c5]);
      const vy = (this.nodePos[c2 + 1] - this.nodePos[c0 + 1]) + (this.nodePos[c3 + 1] - this.nodePos[c1 + 1]) +
                 (this.nodePos[c6 + 1] - this.nodePos[c4 + 1]) + (this.nodePos[c7 + 1] - this.nodePos[c5 + 1]);
      const vz = (this.nodePos[c2 + 2] - this.nodePos[c0 + 2]) + (this.nodePos[c3 + 2] - this.nodePos[c1 + 2]) +
                 (this.nodePos[c6 + 2] - this.nodePos[c4 + 2]) + (this.nodePos[c7 + 2] - this.nodePos[c5 + 2]);

      const wx = (this.nodePos[c4] - this.nodePos[c0]) + (this.nodePos[c5] - this.nodePos[c1]) +
                 (this.nodePos[c6] - this.nodePos[c2]) + (this.nodePos[c7] - this.nodePos[c3]);
      const wy = (this.nodePos[c4 + 1] - this.nodePos[c0 + 1]) + (this.nodePos[c5 + 1] - this.nodePos[c1 + 1]) +
                 (this.nodePos[c6 + 1] - this.nodePos[c2 + 1]) + (this.nodePos[c7 + 1] - this.nodePos[c3 + 1]);
      const wz = (this.nodePos[c4 + 2] - this.nodePos[c0 + 2]) + (this.nodePos[c5 + 2] - this.nodePos[c1 + 2]) +
                 (this.nodePos[c6 + 2] - this.nodePos[c2 + 2]) + (this.nodePos[c7 + 2] - this.nodePos[c3 + 2]);

      // Normalize each column of the cofactor matrix so cell aspect ratios don't skew normals!
      let c0x = vy * wz - vz * wy, c0y = vz * wx - vx * wz, c0z = vx * wy - vy * wx;
      let c1x = wy * uz - wz * uy, c1y = wz * ux - wx * uz, c1z = wx * uy - wy * ux;
      let c2x = uy * vz - uz * vy, c2y = uz * vx - ux * vz, c2z = ux * vy - uy * vx;

      const l0 = 1.0 / (Math.hypot(c0x, c0y, c0z) + 1e-9);
      const l1 = 1.0 / (Math.hypot(c1x, c1y, c1z) + 1e-9);
      const l2 = 1.0 / (Math.hypot(c2x, c2y, c2z) + 1e-9);
      c0x *= l0; c0y *= l0; c0z *= l0;
      c1x *= l1; c1y *= l1; c1z *= l1;
      c2x *= l2; c2y *= l2; c2z *= l2;

      for (let k = 0; k < 8; k++) {
        const mOff = cn[k] * 9;
        this.nodeNormalMat[mOff + 0] += c0x;
        this.nodeNormalMat[mOff + 1] += c1x;
        this.nodeNormalMat[mOff + 2] += c2x;
        this.nodeNormalMat[mOff + 3] += c0y;
        this.nodeNormalMat[mOff + 4] += c1y;
        this.nodeNormalMat[mOff + 5] += c2y;
        this.nodeNormalMat[mOff + 6] += c0z;
        this.nodeNormalMat[mOff + 7] += c1z;
        this.nodeNormalMat[mOff + 8] += c2z;
      }
    }

    // Normalize accumulated normal matrix columns at each node so boundary & interior nodes weight equally
    for (let i = 0; i < this.numNodes; i++) {
      const mOff = i * 9;
      const n0 = Math.hypot(this.nodeNormalMat[mOff + 0], this.nodeNormalMat[mOff + 3], this.nodeNormalMat[mOff + 6]);
      if (n0 > 1e-8) {
        const inv0 = 1.0 / n0;
        const inv1 = 1.0 / (Math.hypot(this.nodeNormalMat[mOff + 1], this.nodeNormalMat[mOff + 4], this.nodeNormalMat[mOff + 7]) + 1e-9);
        const inv2 = 1.0 / (Math.hypot(this.nodeNormalMat[mOff + 2], this.nodeNormalMat[mOff + 5], this.nodeNormalMat[mOff + 8]) + 1e-9);
        this.nodeNormalMat[mOff + 0] *= inv0; this.nodeNormalMat[mOff + 3] *= inv0; this.nodeNormalMat[mOff + 6] *= inv0;
        this.nodeNormalMat[mOff + 1] *= inv1; this.nodeNormalMat[mOff + 4] *= inv1; this.nodeNormalMat[mOff + 7] *= inv1;
        this.nodeNormalMat[mOff + 2] *= inv2; this.nodeNormalMat[mOff + 5] *= inv2; this.nodeNormalMat[mOff + 8] *= inv2;
      }
    }

    // Proximity factor for inter-layer contact ambient occlusion
    const gapProximity = (this.mode === 'sphere')
      ? 1.0
      : Math.max(0.0, 1.0 - (this.explosionGapMm / 24.0));

    // 2. Transform every surface vertex via trilinear weights
    for (let lid = 0; lid < 5; lid++) {
      const range = this.layerDrawRanges[lid];
      const mesh = this.layerMeshes[lid];
      const restNormals = mesh.restNormals;
      const active = this.isLayerActive(lid);

      if (!active) continue;

      for (let v = 0; v < range.vertCount; v++) {
        const gv = range.baseVertex + v;
        const cell = this.hexCells[this.skinCellIdx[gv]];
        const cn = cell.corners;
        const wOff = gv * 8;

        const w0 = this.skinWeights[wOff],     w1 = this.skinWeights[wOff + 1];
        const w2 = this.skinWeights[wOff + 2], w3 = this.skinWeights[wOff + 3];
        const w4 = this.skinWeights[wOff + 4], w5 = this.skinWeights[wOff + 5];
        const w6 = this.skinWeights[wOff + 6], w7 = this.skinWeights[wOff + 7];

        const i0 = cn[0] * 3, i1 = cn[1] * 3, i2 = cn[2] * 3, i3 = cn[3] * 3;
        const i4 = cn[4] * 3, i5 = cn[5] * 3, i6 = cn[6] * 3, i7 = cn[7] * 3;

        const px = w0 * this.nodePos[i0]     + w1 * this.nodePos[i1]     + w2 * this.nodePos[i2]     + w3 * this.nodePos[i3] +
                   w4 * this.nodePos[i4]     + w5 * this.nodePos[i5]     + w6 * this.nodePos[i6]     + w7 * this.nodePos[i7];
        const py = w0 * this.nodePos[i0 + 1] + w1 * this.nodePos[i1 + 1] + w2 * this.nodePos[i2 + 1] + w3 * this.nodePos[i3 + 1] +
                   w4 * this.nodePos[i4 + 1] + w5 * this.nodePos[i5 + 1] + w6 * this.nodePos[i6 + 1] + w7 * this.nodePos[i7 + 1];
        const pz = w0 * this.nodePos[i0 + 2] + w1 * this.nodePos[i1 + 2] + w2 * this.nodePos[i2 + 2] + w3 * this.nodePos[i3 + 2] +
                   w4 * this.nodePos[i4 + 2] + w5 * this.nodePos[i5 + 2] + w6 * this.nodePos[i6 + 2] + w7 * this.nodePos[i7 + 2];

        const rnx = restNormals[v * 3];
        const rny = restNormals[v * 3 + 1];
        const rnz = restNormals[v * 3 + 2];

        // Smoothly interpolate the node-averaged normal matrix across the 8 cell corners
        let nx = 0.0, ny = 0.0, nz = 0.0;
        for (let k = 0; k < 8; k++) {
          const wk = this.skinWeights[wOff + k];
          const mOff = cn[k] * 9;
          nx += wk * (this.nodeNormalMat[mOff + 0] * rnx + this.nodeNormalMat[mOff + 1] * rny + this.nodeNormalMat[mOff + 2] * rnz);
          ny += wk * (this.nodeNormalMat[mOff + 3] * rnx + this.nodeNormalMat[mOff + 4] * rny + this.nodeNormalMat[mOff + 5] * rnz);
          nz += wk * (this.nodeNormalMat[mOff + 6] * rnx + this.nodeNormalMat[mOff + 7] * rny + this.nodeNormalMat[mOff + 8] * rnz);
        }
        const nLen = 1.0 / (Math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9);
        nx *= nLen; ny *= nLen; nz *= nLen;

        // Compute inter-layer contact ambient occlusion from rest-space interface zones
        const vBase = gv * 12;
        const ryLocal = this.skinnedVertexData[vBase + 9];
        const rrLocal = Math.sqrt(px * px + pz * pz);
        let contactOcc = 1.0;
        if (gapProximity > 0.01) {
          if (lid === 0 && ryLocal < 0.14) {
            contactOcc = 1.0 - gapProximity * 0.42 * (1.0 - ryLocal / 0.14);
          } else if (lid === 1) {
            contactOcc = 1.0 - gapProximity * (0.38 * (1.0 - Math.min(1.0, Math.abs(ryLocal) / 0.12)) + 0.25 * (1.0 - rrLocal));
          } else if (lid === 2) {
            contactOcc = 1.0 - gapProximity * (0.35 * Math.min(1.0, Math.abs(ryLocal) / 0.14) + 0.22 * (1.0 - rrLocal));
          } else if (lid === 3) {
            contactOcc = 1.0 - gapProximity * 0.48;
          } else if (lid === 4 && ryLocal > -0.18) {
            contactOcc = 1.0 - gapProximity * 0.42 * (1.0 - Math.abs(ryLocal) / 0.18);
          }
        }

        this.skinnedVertexData[vBase + 0] = px;
        this.skinnedVertexData[vBase + 1] = py;
        this.skinnedVertexData[vBase + 2] = pz;
        this.skinnedVertexData[vBase + 4] = nx;
        this.skinnedVertexData[vBase + 5] = ny;
        this.skinnedVertexData[vBase + 6] = nz;
        this.skinnedVertexData[vBase + 7] = contactOcc;
      }
    }

    // 3. Transform all 9 Snuffle Treats (tucked inside Layer 1 pockets or ejected on studio floor)
    if (this.treats) {
      for (let k = 0; k < this.treats.length; k++) {
        const tr = this.treats[k];
        const range = tr.drawRange;
        const tm = tr.template;
        const rPos = tm.restPositions;
        const rNorm = tm.restNormals;

        if (!this.isTreatVisible(k)) {
          for (let v = 0; v < range.vertCount; v++) {
            const vBase = (range.baseVertex + v) * 12;
            this.skinnedVertexData[vBase + 0] = 0.0;
            this.skinnedVertexData[vBase + 1] = -999.0;
            this.skinnedVertexData[vBase + 2] = 0.0;
          }
          continue;
        }

        if (tr.state === 'hidden') {
          const pWorld = this.getPocketWorldPos(k);
          tr.pos[0] = pWorld[0];
          tr.pos[1] = pWorld[1];
          tr.pos[2] = pWorld[2];
        }

        const cx = Math.cos(tr.euler[0]), sx = Math.sin(tr.euler[0]);
        const cy = Math.cos(tr.euler[1]), sy = Math.sin(tr.euler[1]);
        const cz = Math.cos(tr.euler[2]), sz = Math.sin(tr.euler[2]);

        // Euler Y * X * Z rotation matrix
        let r00 = cy * cz + sy * sx * sz, r01 = -cy * sz + sy * sx * cz, r02 = sy * cx;
        let r10 = cx * sz,                r11 = cx * cz,                 r12 = -sx;
        let r20 = -sy * cz + cy * sx * sz, r21 = sy * sz + cy * sx * cz, r22 = cy * cx;

        // If tucked inside Layer 1, compose with host hex cell rotation so treat tilts with the felt ruffle!
        if (tr.state === 'hidden') {
          const hRot = this.hexCells[tr.hexIdx].rotMat;
          const m00 = hRot[0] * r00 + hRot[1] * r10 + hRot[2] * r20;
          const m01 = hRot[0] * r01 + hRot[1] * r11 + hRot[2] * r21;
          const m02 = hRot[0] * r02 + hRot[1] * r12 + hRot[2] * r22;
          const m10 = hRot[3] * r00 + hRot[4] * r10 + hRot[5] * r20;
          const m11 = hRot[3] * r01 + hRot[4] * r11 + hRot[5] * r21;
          const m12 = hRot[3] * r02 + hRot[4] * r12 + hRot[5] * r22;
          const m20 = hRot[6] * r00 + hRot[7] * r10 + hRot[8] * r20;
          const m21 = hRot[6] * r01 + hRot[7] * r11 + hRot[8] * r21;
          const m22 = hRot[6] * r02 + hRot[7] * r12 + hRot[8] * r22;
          r00 = m00; r01 = m01; r02 = m02;
          r10 = m10; r11 = m11; r12 = m12;
          r20 = m20; r21 = m21; r22 = m22;
        }

        const tx = tr.pos[0], ty = tr.pos[1], tz = tr.pos[2];
        // Subtle pocket occlusion when tucked between green felt ruffles
        const baseAO = (tr.state === 'hidden') ? 0.78 : 0.94;

        for (let v = 0; v < range.vertCount; v++) {
          const vBase = (range.baseVertex + v) * 12;
          const lx = rPos[v * 3], ly = rPos[v * 3 + 1], lz = rPos[v * 3 + 2];
          const nx = rNorm[v * 3], ny = rNorm[v * 3 + 1], nz = rNorm[v * 3 + 2];

          this.skinnedVertexData[vBase + 0] = tx + r00 * lx + r01 * ly + r02 * lz;
          this.skinnedVertexData[vBase + 1] = ty + r10 * lx + r11 * ly + r12 * lz;
          this.skinnedVertexData[vBase + 2] = tz + r20 * lx + r21 * ly + r22 * lz;

          this.skinnedVertexData[vBase + 4] = r00 * nx + r01 * ny + r02 * nz;
          this.skinnedVertexData[vBase + 5] = r10 * nx + r11 * ny + r12 * nz;
          this.skinnedVertexData[vBase + 6] = r20 * nx + r21 * ny + r22 * nz;
          this.skinnedVertexData[vBase + 7] = baseAO;
        }
      }
    }
  }

  // Compute physical Mass (g), Live Deformed Volume (cm^3), Volume Ratio (%), and Kinetic Energy (mJ)
  computeTelemetry() {
    let totalMassG = 0.0;
    let kineticEnergyMJ = 0.0; // 1 mJ = 1e-3 J

    for (let i = 0; i < this.numNodes; i++) {
      if (!this.isLayerActive(this.nodeLayer[i])) continue;
      const mG = this.nodeMass[i];
      totalMassG += mG;

      const idx = i * 3;
      // Convert velocity from world units/s to meters/s: 1 unit = 0.055 m
      const vxM = this.nodeVel[idx] * 0.055;
      const vyM = this.nodeVel[idx + 1] * 0.055;
      const vzM = this.nodeVel[idx + 2] * 0.055;
      const v2 = vxM * vxM + vyM * vyM + vzM * vzM;
      // E_k in mJ: 0.5 * (m_g * 1e-3 kg) * v^2 * 1000 mJ/J = 0.5 * m_g * v^2
      kineticEnergyMJ += 0.5 * mG * v2;
    }

    const cm3PerUnit3 = (MM_PER_UNIT * 0.1) ** 3;
    let currentVolCm3 = 0.0;
    let restVolCm3 = 0.0;

    for (let t = 0; t < this.tets.length; t++) {
      const tet = this.tets[t];
      if (!this.isLayerActive(tet.layerId)) continue;

      const i0 = tet.i0 * 3, i1 = tet.i1 * 3, i2 = tet.i2 * 3, i3 = tet.i3 * 3;
      const x0 = this.nodePos[i0], y0 = this.nodePos[i0 + 1], z0 = this.nodePos[i0 + 2];
      const x10 = this.nodePos[i1] - x0, y10 = this.nodePos[i1 + 1] - y0, z10 = this.nodePos[i1 + 2] - z0;
      const x20 = this.nodePos[i2] - x0, y20 = this.nodePos[i2 + 1] - y0, z20 = this.nodePos[i2 + 2] - z0;
      const x30 = this.nodePos[i3] - x0, y30 = this.nodePos[i3 + 1] - y0, z30 = this.nodePos[i3 + 2] - z0;

      const crossX = y10 * z20 - z10 * y20;
      const crossY = z10 * x20 - x10 * z20;
      const crossZ = x10 * y20 - y10 * x20;
      const vol = Math.abs((crossX * x30 + crossY * y30 + crossZ * z30) / 6.0);

      currentVolCm3 += vol * tet.fillFactor * cm3PerUnit3;
      restVolCm3 += tet.restVol * tet.fillFactor * cm3PerUnit3;
    }

    let hiddenTreatsCount = 0;
    let freeTreatsCount = 0;
    let treatsMassG = 0.0;

    if (this.treats) {
      for (let k = 0; k < this.treats.length; k++) {
        const tr = this.treats[k];
        if (tr.state === 'hidden') {
          hiddenTreatsCount++;
          if (this.mode !== 'sphere') {
            totalMassG += tr.massG;
            treatsMassG += tr.massG;
            currentVolCm3 += tr.volCm3;
            restVolCm3 += tr.volCm3;
          }
        } else if (tr.state === 'free') {
          freeTreatsCount++;
          const vxM = tr.vel[0] * 0.055;
          const vyM = tr.vel[1] * 0.055;
          const vzM = tr.vel[2] * 0.055;
          kineticEnergyMJ += 0.5 * tr.massG * (vxM * vxM + vyM * vyM + vzM * vzM);
        }
      }
    }

    const volRatioPct = restVolCm3 > 0 ? (currentVolCm3 / restVolCm3) * 100.0 : 100.0;

    return {
      massG: totalMassG,
      volumeCm3: currentVolCm3,
      restVolumeCm3: restVolCm3,
      volRatioPct,
      kineticEnergyMJ,
      grabForceN: this.grabState.active ? this.grabState.forceN : 0.0,
      hiddenTreatsCount,
      freeTreatsCount,
      treatsMassG
    };
  }
}
