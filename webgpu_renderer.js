// ============================================================================
// webgpu_renderer.js
// Multi-Pass WebGPU Renderer for the Modular Burger Pet Toy
// Passes:
//   1. 2048x2048 Shadow Map Pass (Directional Key Light)
//   2. Opaque Studio Scene Pass (Cyclorama Backdrop + Floor + Layers 0, 1, 3, 4)
//   3. Copy Opaque Color Texture -> Refraction Background Texture
//   4. Translucent Refractive Pass for Layer 2 (Coral-Red TPE Chew Ring with
//      Beer-Lambert absorption, SSS, Fresnel, thickness translucency & 3D bubbles)
//      + Optional Deformed Mesh Wireframe & XPBD Hexahedral Lattice Overlay
// ============================================================================

import { TOP_SESAME_SEEDS, BOT_SESAME_SEEDS } from './sdf_mesh.js';

// Matrix math helpers (Column-major Float32Array(16) for WebGPU WGSL mat4x4f)
export function mat4Identity() {
  const m = new Float32Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
}

export function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

export function mat4Perspective(fovYRad, aspect, near, far) {
  const f = 1.0 / Math.tan(fovYRad * 0.5);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far / (near - far);
  out[11] = -1.0;
  out[14] = (near * far) / (near - far);
  return out;
}

export function mat4Ortho(left, right, bottom, top, near, far) {
  const out = new Float32Array(16);
  out[0] = 2.0 / (right - left);
  out[5] = 2.0 / (top - bottom);
  out[10] = 1.0 / (near - far);
  out[12] = -(right + left) / (right - left);
  out[13] = -(top + bottom) / (top - bottom);
  out[14] = near / (near - far);
  out[15] = 1.0;
  return out;
}

export function mat4LookAt(eye, target, up) {
  let zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
  let zLen = 1.0 / (Math.sqrt(zx * zx + zy * zy + zz * zz) + 1e-12);
  zx *= zLen; zy *= zLen; zz *= zLen;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * xx;
  // Correct cross product for x = up x z
  xx = up[1] * zz - up[2] * zy;
  xy = up[2] * zx - up[0] * zz;
  xz = up[0] * zy - up[1] * zx;
  let xLen = 1.0 / (Math.sqrt(xx * xx + xy * xy + xz * xz) + 1e-12);
  xx *= xLen; xy *= xLen; xz *= xLen;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  const out = new Float32Array(16);
  out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
  out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
  out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1.0;
  return out;
}

// Invert a 4x4 matrix for screen-to-world raycasting
export function mat4Invert(m) {
  const inv = new Float32Array(16);
  inv[0] = m[5]  * m[10] * m[15] - m[5]  * m[11] * m[14] - m[9]  * m[6]  * m[15] + m[9]  * m[7]  * m[14] + m[13] * m[6]  * m[11] - m[13] * m[7]  * m[10];
  inv[4] = -m[4]  * m[10] * m[15] + m[4]  * m[11] * m[14] + m[8]  * m[6]  * m[15] - m[8]  * m[7]  * m[14] - m[12] * m[6]  * m[11] + m[12] * m[7]  * m[10];
  inv[8] = m[4]  * m[9] * m[15] - m[4]  * m[11] * m[13] - m[8]  * m[5] * m[15] + m[8]  * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4]  * m[9] * m[14] + m[4]  * m[10] * m[13] + m[8]  * m[5] * m[14] - m[8]  * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1]  * m[10] * m[15] + m[1]  * m[11] * m[14] + m[9]  * m[2] * m[15] - m[9]  * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0]  * m[10] * m[15] - m[0]  * m[11] * m[14] - m[8]  * m[2] * m[15] + m[8]  * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0]  * m[9] * m[15] + m[0]  * m[11] * m[13] + m[8]  * m[1] * m[15] - m[8]  * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0]  * m[9] * m[14] - m[0]  * m[10] * m[13] - m[8]  * m[1] * m[14] + m[8]  * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1]  * m[6] * m[15] - m[1]  * m[7] * m[14] - m[5]  * m[2] * m[15] + m[5]  * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0]  * m[6] * m[15] + m[0]  * m[7] * m[14] + m[4]  * m[2] * m[15] - m[4]  * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0]  * m[5] * m[15] - m[0]  * m[7] * m[13] - m[4]  * m[1] * m[15] + m[4]  * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0]  * m[5] * m[14] + m[0]  * m[6] * m[13] + m[4]  * m[1] * m[14] - m[4]  * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

  let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (Math.abs(det) < 1e-12) return mat4Identity();
  det = 1.0 / det;
  for (let i = 0; i < 16; i++) inv[i] *= det;
  return inv;
}

// ============================================================================
// WGSL Shaders (Shadow, Studio Floor, Multi-Material Burger, Wireframe)
// ============================================================================
const WGSL_SHADER_SOURCE = /* wgsl */`
struct Uniforms {
  viewProj     : mat4x4f,
  viewMat      : mat4x4f,
  lightViewProj: mat4x4f,
  cameraPos    : vec4f,   // xyz = eyePos, w = time
  lightDir     : vec4f,   // xyz = normalized light dir (towards light), w = explosionGapMm
  screenParams : vec4f,   // x = width, y = height, z = mode (0:exploded, 1:burger, 2:sphere), w = showMesh
  layerYOffsets: vec4f,   // x=y0, y=y1, z=y2, w=y3
  extraParams  : vec4f,   // x=y4, y=stiffness, z=damping, w=activeMask
};

struct SesameSeed {
  norm   : vec4f, // xyz = normal, w = isTop (1.0 or 0.0)
  uAxis1 : vec4f, // xyz = major axis
  uAxis2 : vec4f, // xyz = minor axis
};

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var shadowMap : texture_depth_2d;
@group(0) @binding(2) var shadowSampler : sampler_comparison;
@group(0) @binding(3) var opaqueSceneTex : texture_2d<f32>;
@group(0) @binding(4) var linearSampler : sampler;
@group(0) @binding(5) var<storage, read> sesameSeeds : array<SesameSeed, 35>;

// ----------------------------------------------------------------------------
// Khronos PBR Neutral Tone Mapping (preserves true CMF hue & saturation)
// ----------------------------------------------------------------------------
fn pbrNeutralToneMapping(colorIn: vec3f) -> vec3f {
  let startCompression = 0.8 - 0.04;
  let desaturation = 0.15;
  var color = max(colorIn, vec3f(0.0));
  let x = min(color.r, min(color.g, color.b));
  let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
  color -= offset;

  let peak = max(color.r, max(color.g, color.b));
  if (peak < startCompression) {
    return color;
  }

  let d = 1.0 - startCompression;
  let newPeak = 1.0 - d * d / (peak + d - startCompression);
  color *= newPeak / peak;

  let g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
  return mix(color, vec3f(newPeak), g);
}

fn linearToSrgb(c: vec3f) -> vec3f {
  return pow(clamp(c, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));
}

// ----------------------------------------------------------------------------
// 3D Procedural Hash & Value Noise (Evaluated in Rest Space vRestPos)
// ----------------------------------------------------------------------------
fn hash31(p: vec3f) -> f32 {
  var p3 = fract(p * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p: vec3f) -> vec3f {
  var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}

fn noise3D(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);

  return mix(
    mix(
      mix(hash31(i + vec3f(0,0,0)), hash31(i + vec3f(1,0,0)), u.x),
      mix(hash31(i + vec3f(0,1,0)), hash31(i + vec3f(1,1,0)), u.x),
      u.y
    ),
    mix(
      mix(hash31(i + vec3f(0,0,1)), hash31(i + vec3f(1,0,1)), u.x),
      mix(hash31(i + vec3f(0,1,1)), hash31(i + vec3f(1,1,1)), u.x),
      u.y
    ),
    u.z
  );
}

// ----------------------------------------------------------------------------
// PCF Soft Shadow Mapping (9-Tap Poisson Filter)
// ----------------------------------------------------------------------------
fn sampleShadowPCF(worldPos: vec3f, worldNormal: vec3f) -> f32 {
  let biasedPos = worldPos + worldNormal * 0.014;
  let lightClip = u.lightViewProj * vec4f(biasedPos, 1.0);
  let ndc = lightClip.xyz / lightClip.w;
  let uv = vec2f(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
  let depth = ndc.z - 0.0025;

  if (uv.x < 0.01 || uv.x > 0.99 || uv.y < 0.01 || uv.y > 0.99 || depth > 0.995) {
    return 1.0;
  }

  let texel = 1.0 / 2048.0;
  let offsets = array<vec2f, 9>(
    vec2f( 0.0,  0.0),
    vec2f(-1.2, -0.8),
    vec2f( 1.2, -0.8),
    vec2f(-0.8,  1.2),
    vec2f( 0.8,  1.2),
    vec2f(-2.1,  0.2),
    vec2f( 2.1, -0.2),
    vec2f( 0.2, -2.1),
    vec2f(-0.2,  2.1)
  );

  var sum = 0.0;
  for (var i = 0u; i < 9u; i = i + 1u) {
    sum += textureSampleCompareLevel(shadowMap, shadowSampler, uv + offsets[i] * texel, depth);
  }
  return mix(0.25, 1.0, sum / 9.0);
}

// ----------------------------------------------------------------------------
// Studio Environment Lighting (Softbox Key, Cool Fill Dome, Warm Bounce)
// ----------------------------------------------------------------------------
fn sampleStudioEnv(dir: vec3f, roughness: f32) -> vec3f {
  let y = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  let skyTop = vec3f(0.95, 0.96, 0.98);
  let horizon = vec3f(0.83, 0.85, 0.88);
  let ground = vec3f(0.68, 0.67, 0.65);
  var env = mix(ground, mix(horizon, skyTop, smoothstep(0.45, 0.95, y)), smoothstep(0.1, 0.55, y));

  // Overhead main softbox reflection panel
  let keyPanelDir = normalize(vec3f(0.42, 0.82, 0.38));
  let rimPanelDir = normalize(vec3f(-0.68, 0.52, -0.50));
  let sidePanelDir = normalize(vec3f(-0.75, 0.25, 0.62));

  let gloss = 1.0 - roughness * 0.75;
  let softbox1 = smoothstep(0.76 - roughness * 0.22, 0.96, dot(dir, keyPanelDir));
  let softbox2 = smoothstep(0.80 - roughness * 0.20, 0.97, dot(dir, rimPanelDir));
  let softbox3 = smoothstep(0.84 - roughness * 0.18, 0.98, dot(dir, sidePanelDir));

  env += vec3f(1.35, 1.32, 1.28) * softbox1 * gloss;
  env += vec3f(0.75, 0.82, 0.92) * softbox2 * gloss;
  env += vec3f(0.55, 0.54, 0.52) * softbox3 * gloss;
  return env;
}

// ----------------------------------------------------------------------------
// Rest-Space Teardrop Sesame Indent Evaluation in Fragment Shader
// Returns xyz = 3D concave normal tilt vector, w = indent depth profile [0..1]
// ----------------------------------------------------------------------------
fn evaluateSesameIndentShader(restPos: vec3f, isTop: bool) -> vec4f {
  let cy = select(restPos.y - 0.04, restPos.y + 0.04, isTop);
  let pSph = vec3f(restPos.x, cy / 0.88, restPos.z);
  let r = length(pSph) + 1e-6;
  let uDir = pSph / r;

  let lenA = select(0.085, 0.095, isTop);
  let widB = select(0.046, 0.052, isTop);
  let startIdx = select(22u, 0u, isTop);
  let endIdx = select(35u, 22u, isTop);

  var maxProfile = 0.0;
  var normDelta = vec3f(0.0);

  for (var i = startIdx; i < endIdx; i = i + 1u) {
    let s = sesameSeeds[i];
    let dotN = dot(uDir, s.norm.xyz);
    if (dotN < 0.952) {
      continue;
    }
    let v = dot(pSph, s.uAxis1.xyz);
    let uCoord = dot(pSph, s.uAxis2.xyz);
    let vn = clamp(v / lenA, -1.2, 1.2);
    let taper = 1.0 - 0.38 * vn;
    let effW = max(0.014, widB * taper);
    let un = uCoord / effW;
    let q2 = un * un + vn * vn;
    if (q2 < 1.0) {
      let f = 1.0 - q2;
      let prof = f * f;
      if (prof > maxProfile) {
        maxProfile = prof;
        // Concave bowl normal tilts inward: gradient of (+indent) has negative sign on (u1*vn + u2*un)
        let slopeFactor = 2.4 * f;
        normDelta = -(s.uAxis1.xyz * vn + s.uAxis2.xyz * un) * slopeFactor;
      }
    }
  }
  return vec4f(normDelta, maxProfile);
}

// ----------------------------------------------------------------------------
// Rest-Space 3D Micro-Bubble Field for Coral-Red Translucent TPE Chew Ring
// ----------------------------------------------------------------------------
fn sampleBubbleLayer(p: vec3f, freq: f32, seedOffset: f32) -> vec4f {
  // Returns xyz = bubble normal in rest space, w = bubble hit intensity [0..1]
  let scaled = p * freq + vec3f(seedOffset);
  let cell = floor(scaled);
  let local = fract(scaled) - vec3f(0.5);
  let rnd = hash33(cell);

  // Sparse ~12% of internal cells contain a tiny suspended air bubble
  if (rnd.x > 0.12) {
    return vec4f(0.0, 1.0, 0.0, 0.0);
  }

  let centerOffset = (rnd - vec3f(0.5)) * 0.44;
  let dVec = local - centerOffset;
  let bubbleRadius = 0.08 + 0.09 * rnd.y; // delicate 0.3 - 0.7 mm micro-bubbles
  let dist = length(dVec);

  if (dist < bubbleRadius) {
    let n = normalize(dVec);
    let radial = dist / bubbleRadius;
    // Bubble optical profile: subtle refractive rim ring + tiny specular pinpoint
    let rim = smoothstep(0.52, 0.94, radial) * (1.0 - smoothstep(0.94, 1.0, radial));
    let core = 1.0 - radial * radial;
    return vec4f(n, clamp(rim * 0.75 + core * 0.35, 0.0, 1.0));
  }
  return vec4f(0.0, 1.0, 0.0, 0.0);
}

// ============================================================================
// PASS 1: Shadow Map Vertex Shader
// ============================================================================
struct VertexIn {
  @location(0) posAndLayer   : vec4f, // xyz = worldPos, w = layerId
  @location(1) normAndAO     : vec4f, // xyz = worldNormal, w = contactAO
  @location(2) restAndThick  : vec4f, // xyz = restPos, w = thickness
};

@vertex
fn vs_shadow(in: VertexIn) -> @builtin(position) vec4f {
  return u.lightViewProj * vec4f(in.posAndLayer.xyz, 1.0);
}

// ============================================================================
// PASS 2A: Studio Cyclorama Backdrop & Shadow-Receiving Ground Plane
// ============================================================================
struct BgOut {
  @builtin(position) pos : vec4f,
  @location(0) uv        : vec2f,
};

@vertex
fn vs_studio_bg(@builtin(vertex_index) vid: u32) -> BgOut {
  // Fullscreen triangle
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0)
  );
  var out: BgOut;
  out.pos = vec4f(pos[vid], 0.9999, 1.0);
  out.uv = pos[vid] * 0.5 + 0.5;
  return out;
}

@fragment
fn fs_studio_bg(in: BgOut) -> @location(0) vec4f {
  // Clean industrial design studio cyclorama backdrop matching the reference photo
  let topColor = vec3f(0.845, 0.855, 0.868);
  let midColor = vec3f(0.815, 0.825, 0.838);
  let botColor = vec3f(0.765, 0.778, 0.792);
  let radial = length((in.uv - vec2f(0.5, 0.52)) * vec2f(1.1, 0.9));
  var bg = mix(topColor, midColor, smoothstep(0.0, 0.85, 1.0 - in.uv.y));
  bg = mix(bg, botColor, smoothstep(0.35, 1.05, radial));
  return vec4f(linearToSrgb(pbrNeutralToneMapping(bg)), 1.0);
}

// Ground Plane Shader (at y = -1.035) to catch real-time shadow map & contact AO
struct FloorOut {
  @builtin(position) pos : vec4f,
  @location(0) worldPos  : vec3f,
};

@vertex
fn vs_floor(@builtin(vertex_index) vid: u32) -> FloorOut {
  var quad = array<vec2f, 6>(
    vec2f(-6.0, -6.0), vec2f( 6.0, -6.0), vec2f( 6.0,  6.0),
    vec2f(-6.0, -6.0), vec2f( 6.0,  6.0), vec2f(-6.0,  6.0)
  );
  let p = quad[vid];
  let floorY = min(-1.036, u.extraParams.x - 0.855);
  let worldPos = vec3f(p.x, floorY, p.y);
  var out: FloorOut;
  out.pos = u.viewProj * vec4f(worldPos, 1.0);
  out.worldPos = worldPos;
  return out;
}

@fragment
fn fs_floor(in: FloorOut) -> @location(0) vec4f {
  let r = length(in.worldPos.xz);
  if (r > 5.5) {
    discard;
  }

  let shadow = sampleShadowPCF(in.worldPos, vec3f(0.0, 1.0, 0.0));
  // Soft analytical contact ambient occlusion directly under the bottom shell
  let floorY = min(-1.036, u.extraParams.x - 0.855);
  let botY = u.extraParams.x - 0.84;
  let distToFloor = max(0.0, botY - floorY);
  let heightAtten = exp(-distToFloor * 1.8);
  let contactRing = smoothstep(1.25, 0.15, r) * heightAtten * 0.42;

  let studioBase = vec3f(0.805, 0.816, 0.830);
  let shadowFactor = mix(1.0, shadow, smoothstep(4.5, 1.2, r)) * (1.0 - contactRing);
  let litFloor = studioBase * mix(0.66, 1.0, shadowFactor);

  // Seamless fade into cyclorama background at radial edges
  let edgeFade = smoothstep(5.2, 2.0, r);
  let finalCol = mix(vec3f(0.79, 0.80, 0.815), litFloor, edgeFade);
  return vec4f(linearToSrgb(pbrNeutralToneMapping(finalCol)), 1.0);
}

// ============================================================================
// PASS 2B & PASS 4: Multi-Material Soft-Body Burger Shader
// ============================================================================
struct BurgerVSOut {
  @builtin(position) clipPos : vec4f,
  @location(0) worldPos      : vec3f,
  @location(1) worldNormal   : vec3f,
  @location(2) restPos       : vec3f,
  @location(3) @interpolate(flat) layerId : i32,
  @location(4) contactAO     : f32,
  @location(5) thickness     : f32,
};

@vertex
fn vs_burger(in: VertexIn) -> BurgerVSOut {
  var out: BurgerVSOut;
  let worldPos = in.posAndLayer.xyz;
  out.clipPos = u.viewProj * vec4f(worldPos, 1.0);
  out.worldPos = worldPos;
  out.worldNormal = normalize(in.normAndAO.xyz);
  out.restPos = in.restAndThick.xyz;
  out.layerId = i32(round(in.posAndLayer.w));
  out.contactAO = in.normAndAO.w;
  out.thickness = in.restAndThick.w;
  return out;
}

// Smith-GGX Microfacet Helper Functions
fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (3.14159265 * denom * denom + 1e-6);
}

fn V_SmithGGXCorrelated(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
  let a2 = roughness * roughness * roughness * roughness;
  let ggxV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
  let ggxL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
  return 0.5 / (ggxV + ggxL + 1e-6);
}

fn F_Schlick(cosTheta: f32, f0: vec3f) -> vec3f {
  return f0 + (vec3f(1.0) - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Charlie Cloth / Felt Sheen Microfacet Distribution (for Layer 1 Matcha Snuffle Felt)
fn D_Charlie(NdotH: f32, sheenRoughness: f32) -> f32 {
  let invR = 1.0 / max(sheenRoughness, 0.04);
  let cos2h = NdotH * NdotH;
  let sin2h = max(1.0 - cos2h, 0.0001);
  return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * 3.14159265);
}

//Orthonormal tangent frame builder from normal
fn buildTangentFrame(N: vec3f, outT: ptr<function, vec3f>, outB: ptr<function, vec3f>) {
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(N.y) > 0.92);
  *outT = normalize(cross(up, N));
  *outB = cross(N, *outT);
}

@fragment
fn fs_burger_opaque(in: BurgerVSOut, @builtin(front_facing) isFront: bool) -> @location(0) vec4f {
  let lid = in.layerId;
  var N = normalize(in.worldNormal);
  if (!isFront) {
    N = -N;
  }

  let V = normalize(u.cameraPos.xyz - in.worldPos);
  let camDist = length(u.cameraPos.xyz - in.worldPos);
  let NdotV_raw = max(dot(N, V), 0.001);

  var T: vec3f;
  var B: vec3f;
  buildTangentFrame(N, &T, &B);

  let L = normalize(u.lightDir.xyz);
  let fillL = normalize(vec3f(-0.65, 0.35, 0.65));
  let backL = normalize(vec3f(0.15, -0.45, -0.88));

  let shadow = sampleShadowPCF(in.worldPos, N);
  let ao = clamp(in.contactAO, 0.35, 1.0);

  var colorLinear = vec3f(0.0);

  // ==========================================================================
  // LAYER 0 & LAYER 4: Cream-Yellow Top & Bottom Shells
  // Matte Silicone GGX + Subsurface Scattering + Teardrop Indents + Fine Grain
  // ==========================================================================
  if (lid == 0 || lid == 4) {
    // 1. Evaluate Fine Matte Mold Grain in Rest-Space SDF (fading at grazing angles & distance!)
    let distFade = smoothstep(6.2, 2.2, camDist);
    let grazingFade = smoothstep(0.08, 0.48, NdotV_raw);
    let grainFade = distFade * grazingFade;

    let gp = in.restPos * 92.0;
    let gx = noise3D(gp + vec3f(0.18, 0.0, 0.0)) - noise3D(gp - vec3f(0.18, 0.0, 0.0));
    let gz = noise3D(gp + vec3f(0.0, 0.0, 0.18)) - noise3D(gp - vec3f(0.0, 0.0, 0.18));

    // 2. Evaluate Rest-Space Teardrop Sesame Indents (only on outer shell wall)
    let isInnerWall = dot(normalize(in.restPos - vec3f(0.0, select(-0.2, 0.2, lid == 0), 0.0)), in.worldNormal) < -0.15;
    var indentMask = 0.0;
    var sesameDelta = vec3f(0.0);
    if (!isInnerWall) {
      let sesame = evaluateSesameIndentShader(in.restPos, lid == 0);
      indentMask = sesame.w;
      sesameDelta = sesame.xyz;
    }

    // Perturb normal with faded fine matte silicone grain + sculptural 3D teardrop indent curvature
    let grainStrength = 0.045 * grainFade * (1.0 - indentMask * 0.65);
    N = normalize(N + T * (gx * grainStrength) + B * (gz * grainStrength) + sesameDelta * 0.34);

    let NdotV = max(dot(N, V), 0.001);
    let NdotL = max(dot(N, L), 0.0);
    let H = normalize(V + L);
    let NdotH = max(dot(N, H), 0.0);
    let VdotH = max(dot(V, H), 0.0);

    // Cream-yellow silicone base color (slightly deeper warm cream inside teardrop indents & inner cavity)
    var baseCream = vec3f(0.89, 0.77, 0.49); // Linearized #EFE0B4 warm butter-cream
    if (indentMask > 0.01) {
      baseCream = mix(baseCream, vec3f(0.76, 0.62, 0.34), indentMask * 0.68);
    }
    if (isInnerWall) {
      baseCream *= vec3f(0.84, 0.78, 0.70);
    }

    // Roughness: matte silicone (0.46), smoother satin finish inside teardrop sesame indents (0.28)
    let roughness = mix(0.46, 0.28, indentMask);
    let f0 = vec3f(0.036); // Silicone IOR ~ 1.47

    // Smith-GGX Specular
    let D = D_GGX(NdotH, roughness);
    let V_vis = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
    let F = F_Schlick(VdotH, f0);
    let specKey = D * V_vis * F * NdotL * shadow * 1.15;

    // Silicone Subsurface Scattering (Wrap Diffuse + Warm Sub-Dermal Transmission)
    let wrap = 0.38;
    let wrapDiff = max(0.0, (dot(N, L) + wrap) / ((1.0 + wrap) * (1.0 + wrap)));
    let sssTint = vec3f(0.98, 0.58, 0.26);
    let backTrans = pow( clamp(dot(-L, N) * 0.65 + 0.35, 0.0, 1.0), 2.2 ) * (1.0 - in.thickness * 0.55);
    let sssContrib = sssTint * (backTrans * 0.28 + smoothstep(0.0, 0.45, wrapDiff) * (1.0 - NdotL) * 0.16);

    let fillDiff = max(0.0, dot(N, fillL) * 0.5 + 0.5) * 0.28;
    let envRefl = sampleStudioEnv(reflect(-V, N), roughness) * F_Schlick(NdotV, f0) * ao;
    let envDiff = sampleStudioEnv(N, 0.92) * 0.42 * ao;

    let diffuse = baseCream * (wrapDiff * shadow * 0.88 + fillDiff + envDiff) + sssContrib * baseCream;
    colorLinear = diffuse + specKey + envRefl * (0.65 + 0.35 * indentMask);
  }

  // ==========================================================================
  // LAYER 1: Matcha-Green Wavy Ruffled Snuffle Fabric Layer
  // Sheen / Fabric Microfacet Shading + Rest-Space 3D Entangled Felt Fiber Noise
  // ==========================================================================
  else if (lid == 1) {
    // 1. Multi-scale 3D Entangled Felt Fiber Noise in Rest-Space SDF
    let fp1 = in.restPos * 55.0;
    let fp2 = in.restPos * 145.0;
    let warp = vec3f(noise3D(fp1), noise3D(fp1 + 19.3), noise3D(fp1 + 41.7)) * 0.65;
    let fiberA = noise3D(fp1 * vec3f(1.8, 0.8, 1.8) + warp);
    let fiberB = noise3D(fp2 + warp * 2.2);
    let fiberVal = fiberA * 0.65 + fiberB * 0.35;

    // Felt micro-normal perturbation
    let dfx = noise3D(fp2 + vec3f(0.22, 0.0, 0.0)) - fiberB;
    let dfz = noise3D(fp2 + vec3f(0.0, 0.0, 0.22)) - fiberB;
    let fiberFade = smoothstep(6.8, 1.8, camDist);
    N = normalize(N + (T * dfx + B * dfz) * 0.20 * fiberFade);

    let NdotV = max(dot(N, V), 0.001);
    let NdotL = max(dot(N, L), 0.0);
    let H = normalize(V + L);
    let NdotH = max(dot(N, H), 0.0);

    // Matcha-green felt color with natural wool/polyester fiber mottling & fold valley self-occlusion
    let matchaDark = vec3f(0.11, 0.19, 0.055);
    let matchaBase = vec3f(0.18, 0.29, 0.088);
    let matchaHigh = vec3f(0.28, 0.41, 0.145);
    var feltAlbedo = mix(matchaDark, matchaBase, smoothstep(0.22, 0.68, fiberVal));
    feltAlbedo = mix(feltAlbedo, matchaHigh, smoothstep(0.68, 0.95, fiberVal));

    // Deepen ruffle fold valleys using rest-space vertical wave coordinate
    let foldAO = smoothstep(-0.14, 0.08, in.restPos.y) * 0.28 + 0.72;
    feltAlbedo *= foldAO * ao;

    // Oren-Nayar / Wrap rough cloth diffuse
    let wrapDiff = max(0.0, (dot(N, L) + 0.42) / 1.42);
    let fillDiff = max(0.0, dot(N, fillL) * 0.5 + 0.5) * 0.30;

    // Charlie Cloth Microfacet Sheen + Grazing Velvet Fuzz Rim
    let sheenColor = vec3f(0.54, 0.68, 0.34);
    let sheenRoughness = 0.52;
    let charlieD = D_Charlie(NdotH, sheenRoughness);
    let sheenVis = 1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV) + 1e-5);
    let sheenSpec = sheenColor * charlieD * sheenVis * NdotL * shadow * 0.36;

    // Soft grazing felt fuzz halo
    let grazingFuzz = pow(1.0 - NdotV, 3.2) * sheenColor * (0.24 + 0.22 * shadow) * ao;

    // Thin-felt translucent leaf scatter when backlit
    let feltTrans = max(0.0, dot(-L, N) * 0.5 + 0.5) * vec3f(0.32, 0.48, 0.14) * 0.14 * shadow;

    let envLight = sampleStudioEnv(N, 0.95) * 0.44 * ao;
    colorLinear = feltAlbedo * (wrapDiff * shadow * 0.92 + fillDiff + envLight) + sheenSpec + grazingFuzz + feltTrans;
  }

  // ==========================================================================
  // LAYER 3: Warm-Orange Frisbee Disc Module with a Recessed Rim
  // Matte Silicone GGX + Warm Elastomer Subsurface Scattering
  // ==========================================================================
  else if (lid == 3) {
    // Rest-space SDF radial rim detail & fine elastomer grain
    let rRest = length(in.restPos.xz);
    let rimStepMask = exp(-pow((rRest - 0.680) / 0.035, 2.0));
    let distFade = smoothstep(6.5, 2.0, camDist) * smoothstep(0.08, 0.45, NdotV_raw);
    let gp = in.restPos * 85.0;
    let gx = noise3D(gp + vec3f(0.15, 0.0, 0.0)) - noise3D(gp - vec3f(0.15, 0.0, 0.0));
    let gz = noise3D(gp + vec3f(0.0, 0.0, 0.15)) - noise3D(gp - vec3f(0.0, 0.0, 0.15));
    N = normalize(N + (T * gx + B * gz) * 0.038 * distFade);

    let NdotV = max(dot(N, V), 0.001);
    let NdotL = max(dot(N, L), 0.0);
    let H = normalize(V + L);
    let NdotH = max(dot(N, H), 0.0);
    let VdotH = max(dot(V, H), 0.0);

    // Warm orange elastomer (#EA8238) with subtle recessed-well cavity shading
    let wellMask = smoothstep(0.71, 0.64, rRest) * smoothstep(0.0, 0.05, in.restPos.y);
    var orangeBase = vec3f(0.84, 0.26, 0.048);
    orangeBase = mix(orangeBase, vec3f(0.75, 0.21, 0.035), wellMask * 0.18 + rimStepMask * 0.14);

    let roughness = mix(0.42, 0.34, rimStepMask);
    let f0 = vec3f(0.038);

    let D = D_GGX(NdotH, roughness);
    let V_vis = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
    let F = F_Schlick(VdotH, f0);
    let spec = D * V_vis * F * NdotL * shadow * 1.1;

    // Warm tangerine elastomer SSS
    let wrapDiff = max(0.0, (dot(N, L) + 0.42) / (1.42 * 1.42));
    let sssGlow = vec3f(1.0, 0.36, 0.06) * pow(clamp(dot(-L, N) * 0.5 + 0.5, 0.0, 1.0), 2.0) * (1.0 - in.thickness * 0.5) * 0.26;
    let fillDiff = max(0.0, dot(N, fillL) * 0.5 + 0.5) * 0.26;
    let envRefl = sampleStudioEnv(reflect(-V, N), roughness) * F_Schlick(NdotV, f0) * ao;
    let envDiff = sampleStudioEnv(N, 0.90) * 0.40 * ao;

    colorLinear = orangeBase * (wrapDiff * shadow * 0.90 + fillDiff + envDiff) + sssGlow + spec + envRefl * 0.72;
  }

  // ==========================================================================
  // LAYERS 5, 6, 7: Snuffle Pet Treats (Egg Yolk Cube, Salmon Bite, Star Kibble)
  // Rest-Space 3D Porous Freeze-Dried & Baked Biscuit Micro-Texture + SSS
  // ==========================================================================
  else if (lid >= 5) {
    let tp = in.restPos * 95.0;
    let n1 = noise3D(tp);
    let n2 = noise3D(tp * 2.3 + vec3f(17.1, 31.4, 11.8));
    let crumbVal = n1 * 0.6 + n2 * 0.4;

    // Porous freeze-dried / baked crumb normal perturbation
    let dnx = noise3D(tp * 1.8 + vec3f(0.24, 0.0, 0.0)) - n1;
    let dnz = noise3D(tp * 1.8 + vec3f(0.0, 0.0, 0.24)) - n1;
    N = normalize(N + (T * dnx + B * dnz) * 0.24);

    let NdotV = max(dot(N, V), 0.001);
    let NdotL = max(dot(N, L), 0.0);
    let H = normalize(V + L);
    let NdotH = max(dot(N, H), 0.0);
    let VdotH = max(dot(V, H), 0.0);

    var treatBase = vec3f(0.88, 0.50, 0.09);
    var sssColor = vec3f(1.0, 0.62, 0.14);
    var roughness = 0.64;

    if (lid == 5) {
      // Freeze-Dried Egg Yolk / Cheddar Cube (#F2A93B)
      let yolkLight = vec3f(0.96, 0.64, 0.16);
      let yolkDeep  = vec3f(0.82, 0.40, 0.06);
      treatBase = mix(yolkDeep, yolkLight, smoothstep(0.25, 0.80, crumbVal));
      sssColor = vec3f(1.0, 0.68, 0.18);
      roughness = 0.66;
    } else if (lid == 6) {
      // Freeze-Dried Salmon / Meat Bite (#DF6D53) with creamy marbling streaks
      let streak = sin(in.restPos.x * 85.0 + in.restPos.z * 55.0 + n1 * 3.5);
      let salmonMeat = vec3f(0.78, 0.23, 0.13);
      let salmonFat  = vec3f(0.95, 0.68, 0.52);
      treatBase = mix(salmonMeat, salmonFat, smoothstep(0.55, 0.92, streak) * 0.55 + crumbVal * 0.18);
      sssColor = vec3f(0.96, 0.35, 0.20);
      roughness = 0.58;
    } else {
      // Crunchy 5-Lobed Star Kibble Biscuit (#BD7438) with toasted crust edge
      let rStar = length(in.restPos.xz);
      let toastMask = smoothstep(0.03, 0.075, rStar);
      let biscuitCore  = vec3f(0.68, 0.36, 0.12);
      let biscuitCrust = vec3f(0.42, 0.18, 0.045);
      treatBase = mix(biscuitCore, biscuitCrust, toastMask * 0.65 + (1.0 - crumbVal) * 0.25);
      sssColor = vec3f(0.85, 0.42, 0.12);
      roughness = 0.54;
    }

    let f0 = vec3f(0.032);
    let D = D_GGX(NdotH, roughness);
    let V_vis = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
    let F = F_Schlick(VdotH, f0);
    let spec = D * V_vis * F * NdotL * shadow * 0.85;

    let wrapDiff = max(0.0, (dot(N, L) + 0.45) / (1.45 * 1.45));
    let fillDiff = max(0.0, dot(N, fillL) * 0.5 + 0.5) * 0.28;
    let envDiff = sampleStudioEnv(N, 0.92) * 0.42 * ao;
    let sss = sssColor * pow(clamp(dot(-L, N) * 0.5 + 0.5, 0.0, 1.0), 2.0) * 0.18;

    colorLinear = treatBase * (wrapDiff * shadow * 0.92 + fillDiff + envDiff) + sss + spec;
  }

  return vec4f(linearToSrgb(pbrNeutralToneMapping(colorLinear)), 1.0);
}

// ============================================================================
// PASS 4: Translucent Coral-Red TPE Chew Ring (Layer 2)
// Screen-Space Refraction + Beer-Lambert Volumetric Absorption + SSS + Fresnel
// + Thickness-Based Translucency + Rest-Space 3D Embedded Micro Air Bubbles
// ============================================================================
@fragment
fn fs_burger_translucent(in: BurgerVSOut, @builtin(front_facing) isFront: bool) -> @location(0) vec4f {
  var N = normalize(in.worldNormal);
  if (!isFront) {
    N = -N;
  }

  let V = normalize(u.cameraPos.xyz - in.worldPos);
  let L = normalize(u.lightDir.xyz);
  let fillL = normalize(vec3f(-0.65, 0.35, 0.65));

  var T: vec3f;
  var B: vec3f;
  buildTangentFrame(N, &T, &B);

  // 12 rounded outer bumps modulation from rest-space SDF
  let rRest = length(in.restPos.xz);
  let thetaRest = atan2(in.restPos.z, in.restPos.x);
  let bumpCos = cos(12.0 * thetaRest);
  let bumpSin = sin(12.0 * thetaRest);
  let bumpWave = 0.5 + 0.5 * bumpCos;
  let outerBelt = smoothstep(0.75, 0.88, rRest) * smoothstep(0.14, 0.02, abs(in.restPos.y));

  // Accentuate the 12 rounded outer cushion bumps & vertical grooves in normal space
  let azimTangent = normalize(vec3f(-in.restPos.z, 0.0, in.restPos.x));
  N = normalize(N + azimTangent * (bumpSin * outerBelt * 0.22));

  let NdotV = max(dot(N, V), 0.001);
  let NdotL = max(dot(N, L), 0.0);
  let H = normalize(V + L);
  let NdotH = max(dot(N, H), 0.0);
  let VdotH = max(dot(V, H), 0.0);

  let shadow = sampleShadowPCF(in.worldPos, N);
  let ao = clamp(in.contactAO, 0.45, 1.0);

  // --------------------------------------------------------------------------
  // 1. Optical Path Length & Rest-Space Bump/Groove Thickness Profile
  // --------------------------------------------------------------------------
  let opticalThickness = clamp(in.thickness * (0.76 + 0.34 * (1.0 - bumpWave) * outerBelt), 0.16, 1.15);
  let pathLength = opticalThickness * (0.65 + 0.48 / max(0.24, NdotV));

  // --------------------------------------------------------------------------
  // 2. Screen-Space Refraction with Chromatic Dispersion
  // --------------------------------------------------------------------------
  let screenUV = in.clipPos.xy / u.screenParams.xy;
  let viewNormal = (u.viewMat * vec4f(N, 0.0)).xyz;
  let refractStrength = 0.048 * (0.6 + 0.55 * opticalThickness);
  let refractOffset = viewNormal.xy * vec2f(-refractStrength, refractStrength);

  let uvR = clamp(screenUV + refractOffset * 0.94, vec2f(0.002), vec2f(0.998));
  let uvG = clamp(screenUV + refractOffset * 1.00, vec2f(0.002), vec2f(0.998));
  let uvB = clamp(screenUV + refractOffset * 1.06, vec2f(0.002), vec2f(0.998));

  // Sample opaque scene texture (convert from sRGB back to linear for physical Beer-Lambert math)
  let bgR = pow(textureSampleLevel(opaqueSceneTex, linearSampler, uvR, 0.0).r, 2.2);
  let bgG = pow(textureSampleLevel(opaqueSceneTex, linearSampler, uvG, 0.0).g, 2.2);
  let bgB = pow(textureSampleLevel(opaqueSceneTex, linearSampler, uvB, 0.0).b, 2.2);
  let refractedScene = vec3f(bgR, bgG, bgB);

  // --------------------------------------------------------------------------
  // 3. Beer-Lambert Volumetric Absorption: T(d) = exp(-sigma_a * d)
  // --------------------------------------------------------------------------
  // Coral-red TPE strongly absorbs blue & green while transmitting warm coral-red/orange
  let sigmaA = vec3f(0.32, 2.45, 3.65);
  let transmittance = exp(-sigmaA * pathLength * 1.15);

  // --------------------------------------------------------------------------
  // 4. Subsurface Scattering (SSS) & Internal Translucent Jelly Glow
  // --------------------------------------------------------------------------
  let coralScatterAlbedo = vec3f(0.90, 0.21, 0.12); // Juicy coral-red body
  let coralWarmCore = vec3f(1.00, 0.44, 0.22);      // Bright salmon-orange bump center glow
  let grooveShade = 1.0 - (1.0 - bumpWave) * outerBelt * 0.28;

  let wrapL = max(0.0, (dot(N, L) + 0.55) / 1.55);
  let backLight = pow(clamp(dot(-L, V) * 0.5 + dot(-L, N) * 0.5 + 0.35, 0.0, 1.0), 1.8);
  let thinGlow = exp(-opticalThickness * 1.35);

  let internalSSS = mix(coralScatterAlbedo, coralWarmCore, thinGlow * 0.55 + bumpWave * outerBelt * 0.42) *
                    (wrapL * shadow * 0.76 + backLight * 0.48 + 0.28) * grooveShade * ao;

  // Combine refracted background filtered by Beer-Lambert absorption with internal TPE volume scattering
  let milkyTurbidity = clamp(0.56 + 0.24 * opticalThickness, 0.50, 0.82);
  var bodyColor = mix(refractedScene * transmittance, internalSSS, milkyTurbidity) +
                  refractedScene * transmittance * (1.0 - milkyTurbidity) * 0.42;

  // --------------------------------------------------------------------------
  // 5. Tiny Embedded Air Bubbles in Rest-Space 3D SDF (Parallax Ray-Sampled)
  // --------------------------------------------------------------------------
  let interiorMask = smoothstep(0.22, 0.48, in.thickness);
  if (interiorMask > 0.01) {
    let refractRayRest = normalize(in.restPos - u.cameraPos.xyz * 0.15 - N * 0.35);
    var bubbleAccum = vec3f(0.0);

    // 3 parallax depth slices inside the TPE ring volume
    for (var slice = 1; slice <= 3; slice = slice + 1) {
      let depthStep = f32(slice) * 0.022;
      let samplePos = in.restPos + refractRayRest * depthStep;
      let b1 = sampleBubbleLayer(samplePos, 26.0, f32(slice) * 11.7);
      let b2 = sampleBubbleLayer(samplePos, 42.0, f32(slice) * 23.4 + 5.1);

      let bWeight = max(b1.w, b2.w * 0.80) * interiorMask * (1.15 - f32(slice) * 0.24);
      if (bWeight > 0.01) {
        let bNorm = select(b2.xyz, b1.xyz, b1.w >= b2.w * 0.80);
        let bSpec = pow(max(dot(bNorm, H), 0.0), 32.0);
        let bRim = pow(1.0 - abs(dot(bNorm, V)), 2.0);
        let bubbleCol = vec3f(1.0, 0.86, 0.78) * (bRim * 0.50 + bSpec * 1.15 + 0.20);
        bubbleAccum += bubbleCol * bWeight * 0.38;
      }
    }
    bodyColor += bubbleAccum;
  }

  // --------------------------------------------------------------------------
  // 6. Dielectric Fresnel (IOR = 1.49) & Satin-Gloss TPE Specular Reflection
  // --------------------------------------------------------------------------
  let f0 = vec3f(0.039); // TPE IOR = 1.49
  let F = F_Schlick(NdotV, f0);

  // Soft glossy TPE highlight matching the frosty-gloss look in the reference photo
  let D1 = D_GGX(NdotH, 0.25);
  let V1 = V_SmithGGXCorrelated(NdotV, NdotL, 0.25);
  let D2 = D_GGX(NdotH, 0.45);
  let V2 = V_SmithGGXCorrelated(NdotV, NdotL, 0.45);
  let specDirect = (D1 * V1 * 0.55 + D2 * V2 * 0.45) * F_Schlick(VdotH, f0) * NdotL * shadow * 1.15;

  let envRefl = sampleStudioEnv(reflect(-V, N), 0.24) * F;
  // Soft translucent frosty rim sheen on the outer rounded bumps & inner collar (visible in reference photo!)
  let frostyRim = pow(1.0 - NdotV, 2.4) * vec3f(1.0, 0.78, 0.70) * (0.28 + 0.20 * shadow) * ao;

  let finalLinear = bodyColor * (1.0 - F * 0.55) + specDirect + envRefl * 0.75 + frostyRim;
  return vec4f(linearToSrgb(pbrNeutralToneMapping(finalLinear)), 1.0);
}

// ============================================================================
// Wireframe & XPBD Lattice Overlay Shader ("Show Mesh")
// ============================================================================
struct LineIn {
  @location(0) pos : vec3f,
  @location(1) col : vec4f,
};

struct LineOut {
  @builtin(position) clipPos : vec4f,
  @location(0) col           : vec4f,
};

@vertex
fn vs_lines(in: LineIn) -> LineOut {
  var out: LineOut;
  // Slight depth bias toward camera so wireframe sits cleanly atop surface
  let toCam = normalize(u.cameraPos.xyz - in.pos);
  let biasedPos = in.pos + toCam * 0.004;
  out.clipPos = u.viewProj * vec4f(biasedPos, 1.0);
  out.col = in.col;
  return out;
}

@fragment
fn fs_lines(in: LineOut) -> @location(0) vec4f {
  return in.col;
}
`;

export class BurgerWebGPURenderer {
  constructor(canvas, solver) {
    this.canvas = canvas;
    this.solver = solver;
    this.device = null;
    this.context = null;
    this.format = 'bgra8unorm';
    this.showMesh = false;
    this.width = 0;
    this.height = 0;
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not supported in this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
                 || await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("Failed to acquire a WebGPU adapter.");
    }
    this.device = await adapter.requestDevice();
    this.context = this.canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();

    this.context.configure({
      device: this.device,
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      alphaMode: 'opaque'
    });

    this._createBuffers();
    this._createShadowResources();
    await this._createPipelines();
    this.resize(this.canvas.clientWidth || 960, this.canvas.clientHeight || 640);
  }

  _createBuffers() {
    const dev = this.device;

    // Uniform buffer (320 bytes aligned)
    this.uniformData = new Float32Array(80);
    this.uniformBuffer = dev.createBuffer({
      size: 320,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Sesame seed storage buffer (22 top + 13 bottom = 35 seeds * 12 floats * 4 bytes = 1680 bytes)
    const allSeeds = [...TOP_SESAME_SEEDS, ...BOT_SESAME_SEEDS];
    const seedFloats = new Float32Array(35 * 12);
    for (let i = 0; i < Math.min(35, allSeeds.length); i++) {
      const s = allSeeds[i];
      const off = i * 12;
      seedFloats[off + 0] = s.nx;  seedFloats[off + 1] = s.ny;  seedFloats[off + 2] = s.nz;  seedFloats[off + 3] = i < TOP_SESAME_SEEDS.length ? 1.0 : 0.0;
      seedFloats[off + 4] = s.u1x; seedFloats[off + 5] = s.u1y; seedFloats[off + 6] = s.u1z; seedFloats[off + 7] = 0.0;
      seedFloats[off + 8] = s.u2x; seedFloats[off + 9] = s.u2y; seedFloats[off + 10]= s.u2z; seedFloats[off + 11]= 0.0;
    }
    this.sesameBuffer = dev.createBuffer({
      size: seedFloats.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    dev.queue.writeBuffer(this.sesameBuffer, 0, seedFloats);

    // Main Skinned Surface Mesh Vertex Buffer & Index Buffer
    this.vertexBuffer = dev.createBuffer({
      size: this.solver.skinnedVertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    dev.queue.writeBuffer(this.vertexBuffer, 0, this.solver.skinnedVertexData);

    this.indexBuffer = dev.createBuffer({
      size: this.solver.combinedIndices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    });
    dev.queue.writeBuffer(this.indexBuffer, 0, this.solver.combinedIndices);

    // Build decimated surface wireframe + XPBD lattice line buffer for "Show Mesh" mode
    const maxLineVerts = this.solver.latticeLineIndices.length + 24000;
    this.lineVertexData = new Float32Array(maxLineVerts * 7); // [x,y,z, r,g,b,a]
    this.lineVertexBuffer = dev.createBuffer({
      size: this.lineVertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
  }

  _createShadowResources() {
    const dev = this.device;
    this.shadowTexture = dev.createTexture({
      size: [2048, 2048, 1],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    });
    this.shadowView = this.shadowTexture.createView();

    this.shadowSampler = dev.createSampler({
      compare: 'less',
      magFilter: 'linear',
      minFilter: 'linear'
    });

    this.linearSampler = dev.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge'
    });
  }

  async _createPipelines() {
    const dev = this.device;
    const shaderModule = dev.createShaderModule({ code: WGSL_SHADER_SOURCE });
    if (shaderModule.getCompilationInfo) {
      const info = await shaderModule.getCompilationInfo();
      for (const msg of info.messages) {
        if (msg.type === 'error') {
          throw new Error(`WGSL Error (line ${msg.lineNum}): ${msg.message}`);
        }
      }
    }

    // Shadow pass only needs uniform buffer at binding 0 (avoids binding shadowView while rendering to it)
    this.shadowBindGroupLayout = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
      ]
    });

    this.shadowBindGroup = dev.createBindGroup({
      layout: this.shadowBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } }
      ]
    });

    const shadowPipelineLayout = dev.createPipelineLayout({
      bindGroupLayouts: [this.shadowBindGroupLayout]
    });

    this.bindGroupLayout = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
      ]
    });

    const pipelineLayout = dev.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout]
    });

    const vertexBufferLayout = {
      arrayStride: 48, // 12 floats * 4 bytes
      attributes: [
        { shaderLocation: 0, offset: 0,  format: 'float32x4' }, // posAndLayer
        { shaderLocation: 1, offset: 16, format: 'float32x4' }, // normAndAO
        { shaderLocation: 2, offset: 32, format: 'float32x4' }  // restAndThick
      ]
    };

    // 1. Shadow Map Pipeline
    this.shadowPipeline = dev.createRenderPipeline({
      layout: shadowPipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_shadow',
        buffers: [vertexBufferLayout]
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        format: 'depth32float',
        depthWriteEnabled: true,
        depthCompare: 'less'
      }
    });

    // 2. Studio Cyclorama Background Pipeline
    this.bgPipeline = dev.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: 'vs_studio_bg' },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_studio_bg',
        targets: [{ format: this.format }]
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: false,
        depthCompare: 'always'
      }
    });

    // 3. Shadow-Receiving Studio Floor Pipeline
    this.floorPipeline = dev.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: 'vs_floor' },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_floor',
        targets: [{ format: this.format }]
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less-equal'
      }
    });

    // 4. Opaque Layers Pipeline (Layers 0, 1, 3, 4)
    this.opaqueBurgerPipeline = dev.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_burger',
        buffers: [vertexBufferLayout]
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_burger_opaque',
        targets: [{ format: this.format }]
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less'
      }
    });

    // 5. Translucent Refractive Coral-Red TPE Ring Pipeline (Layer 2)
    this.translucentBurgerPipeline = dev.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_burger',
        buffers: [vertexBufferLayout]
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_burger_translucent',
        targets: [{ format: this.format }]
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less'
      }
    });

    // 6. Wireframe / XPBD Lattice Line Overlay Pipeline
    this.linePipeline = dev.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_lines',
        buffers: [{
          arrayStride: 28, // 7 floats * 4 bytes
          attributes: [
            { shaderLocation: 0, offset: 0,  format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x4' }
          ]
        }]
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_lines',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
          }
        }]
      },
      primitive: { topology: 'line-list' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: false,
        depthCompare: 'less-equal'
      }
    });
  }

  resize(width, height) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(64, Math.floor(width * dpr));
    const h = Math.max(64, Math.floor(height * dpr));
    if (w === this.width && h === this.height) return;

    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;

    if (this.depthTexture) this.depthTexture.destroy();
    if (this.opaqueSceneTexture) this.opaqueSceneTexture.destroy();

    this.depthTexture = this.device.createTexture({
      size: [w, h, 1],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    this.depthView = this.depthTexture.createView();

    // Offscreen texture copy of the opaque scene for true refraction inside the Coral-Red TPE ring
    this.opaqueSceneTexture = this.device.createTexture({
      size: [w, h, 1],
      format: this.format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    });
    this.opaqueSceneView = this.opaqueSceneTexture.createView();

    this.mainBindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.shadowView },
        { binding: 2, resource: this.shadowSampler },
        { binding: 3, resource: this.opaqueSceneView },
        { binding: 4, resource: this.linearSampler },
        { binding: 5, resource: { buffer: this.sesameBuffer } }
      ]
    });
  }

  _buildOverlayLines() {
    let ptr = 0;
    const maxFloats = this.lineVertexData.length;

    const pushLine = (ax, ay, az, bx, by, bz, r, g, b, a) => {
      if (ptr + 14 > maxFloats) return;
      this.lineVertexData[ptr++] = ax;
      this.lineVertexData[ptr++] = ay;
      this.lineVertexData[ptr++] = az;
      this.lineVertexData[ptr++] = r;
      this.lineVertexData[ptr++] = g;
      this.lineVertexData[ptr++] = b;
      this.lineVertexData[ptr++] = a;

      this.lineVertexData[ptr++] = bx;
      this.lineVertexData[ptr++] = by;
      this.lineVertexData[ptr++] = bz;
      this.lineVertexData[ptr++] = r;
      this.lineVertexData[ptr++] = g;
      this.lineVertexData[ptr++] = b;
      this.lineVertexData[ptr++] = a;
    };

    // 1. If user is actively grabbing a layer, draw an interactive 3D force tether line
    if (this.solver.grabState.active) {
      const c = this.solver.grabState.currentCenter;
      const t = this.solver.grabState.targetPos;
      pushLine(c[0], c[1], c[2], t[0], t[1], t[2], 1.0, 0.32, 0.22, 0.92);
    }

    // 2. If "Show Mesh" is enabled, draw XPBD Hexahedral Lattice + Surface Mesh Wireframe
    if (this.showMesh) {
      const lIndices = this.solver.latticeLineIndices;
      const nPos = this.solver.nodePos;
      const nLayer = this.solver.nodeLayer;

      // Draw XPBD hexahedral lattice edges
      for (let i = 0; i < lIndices.length; i += 2) {
        const na = lIndices[i], nb = lIndices[i + 1];
        const lid = nLayer[na];
        if (!this.solver.isLayerActive(lid)) continue;
        const a3 = na * 3, b3 = nb * 3;
        pushLine(
          nPos[a3], nPos[a3 + 1], nPos[a3 + 2],
          nPos[b3], nPos[b3 + 1], nPos[b3 + 2],
          0.12, 0.42, 0.85, 0.42
        );
      }

      // Draw decimated surface mesh wireframe triangles
      const vData = this.solver.skinnedVertexData;
      const idxData = this.solver.combinedIndices;
      for (let lid = 0; lid < 5; lid++) {
        if (!this.solver.isLayerActive(lid)) continue;
        const range = this.solver.layerDrawRanges[lid];
        for (let t = 0; t < range.indexCount; t += 9) { // every 3rd triangle for clean visual density
          const i0 = idxData[range.firstIndex + t] * 12;
          const i1 = idxData[range.firstIndex + t + 1] * 12;
          const i2 = idxData[range.firstIndex + t + 2] * 12;
          pushLine(vData[i0], vData[i0+1], vData[i0+2], vData[i1], vData[i1+1], vData[i1+2], 0.10, 0.12, 0.15, 0.26);
          pushLine(vData[i1], vData[i1+1], vData[i1+2], vData[i2], vData[i2+1], vData[i2+2], 0.10, 0.12, 0.15, 0.26);
        }
      }
    }

    const numVerts = ptr / 7;
    if (numVerts > 0) {
      this.device.queue.writeBuffer(this.lineVertexBuffer, 0, this.lineVertexData.buffer, 0, ptr * 4);
    }
    return numVerts;
  }

  render(cameraState, timeSec) {
    if (!this.device) return;
    const dev = this.device;

    // Upload latest skinned vertex positions & normals
    dev.queue.writeBuffer(this.vertexBuffer, 0, this.solver.skinnedVertexData);

    // Compute camera & light matrices
    const aspect = this.width / Math.max(1, this.height);
    const projMat = mat4Perspective(36.0 * Math.PI / 180.0, aspect, 0.15, 28.0);
    const viewMat = mat4LookAt(cameraState.eye, cameraState.target, [0, 1, 0]);
    const viewProj = mat4Multiply(projMat, viewMat);

    // Directional key light (studio upper-front-right angle matching the reference photo's lighting!)
    const lightDir = [0.42, 0.78, 0.46];
    const lLen = Math.hypot(lightDir[0], lightDir[1], lightDir[2]);
    lightDir[0] /= lLen; lightDir[1] /= lLen; lightDir[2] /= lLen;

    const lightEye = [lightDir[0] * 6.0, lightDir[1] * 6.0 + 0.2, lightDir[2] * 6.0];
    const lightView = mat4LookAt(lightEye, [0, 0.1, 0], [0, 1, 0]);
    const lightProj = mat4Ortho(-2.35, 2.35, -2.55, 2.55, 1.0, 12.0);
    const lightViewProj = mat4Multiply(lightProj, lightView);

    // Pack uniforms
    this.uniformData.set(viewProj, 0);
    this.uniformData.set(viewMat, 16);
    this.uniformData.set(lightViewProj, 32);
    this.uniformData[48] = cameraState.eye[0];
    this.uniformData[49] = cameraState.eye[1];
    this.uniformData[50] = cameraState.eye[2];
    this.uniformData[51] = timeSec;

    this.uniformData[52] = lightDir[0];
    this.uniformData[53] = lightDir[1];
    this.uniformData[54] = lightDir[2];
    this.uniformData[55] = this.solver.explosionGapMm;

    const modeCode = this.solver.mode === 'exploded' ? 0.0 : (this.solver.mode === 'burger' ? 1.0 : 2.0);
    this.uniformData[56] = this.width;
    this.uniformData[57] = this.height;
    this.uniformData[58] = modeCode;
    this.uniformData[59] = this.showMesh ? 1.0 : 0.0;

    this.uniformData[60] = this.solver.getLayerTargetY(0);
    this.uniformData[61] = this.solver.getLayerTargetY(1);
    this.uniformData[62] = this.solver.getLayerTargetY(2);
    this.uniformData[63] = this.solver.getLayerTargetY(3);

    this.uniformData[64] = this.solver.getLayerTargetY(4);
    this.uniformData[65] = this.solver.stiffnessScale;
    this.uniformData[66] = this.solver.dampingScale;
    this.uniformData[67] = 1.0;

    dev.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);

    const lineVertCount = this._buildOverlayLines();
    const commandEncoder = dev.createCommandEncoder();

    // ========================================================================
    // PASS 1: Global Shadow Map Pass (2048x2048)
    // ========================================================================
    {
      const shadowPass = commandEncoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.shadowView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store'
        }
      });
      shadowPass.setPipeline(this.shadowPipeline);
      shadowPass.setBindGroup(0, this.shadowBindGroup);
      shadowPass.setVertexBuffer(0, this.vertexBuffer);
      shadowPass.setIndexBuffer(this.indexBuffer, 'uint32');

      for (let lid = 0; lid < 5; lid++) {
        if (!this.solver.isLayerActive(lid)) continue;
        const range = this.solver.layerDrawRanges[lid];
        shadowPass.drawIndexed(range.indexCount, 1, range.firstIndex, 0, 0);
      }
      if (this.solver.treatDrawRanges) {
        for (let k = 0; k < this.solver.treatDrawRanges.length; k++) {
          if (!this.solver.isTreatVisible(k)) continue;
          const trRange = this.solver.treatDrawRanges[k];
          shadowPass.drawIndexed(trRange.indexCount, 1, trRange.firstIndex, 0, 0);
        }
      }
      shadowPass.end();
    }

    const canvasTexture = this.context.getCurrentTexture();
    const canvasView = canvasTexture.createView();

    // ========================================================================
    // PASS 2: Opaque Scene Pass (Studio Cyclorama + Floor + Layers 0, 1, 3, 4 + Treats)
    // ========================================================================
    {
      const opaquePass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: canvasView,
          clearValue: { r: 0.82, g: 0.83, b: 0.85, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store'
        }],
        depthStencilAttachment: {
          view: this.depthView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store'
        }
      });

      opaquePass.setBindGroup(0, this.mainBindGroup);

      // 2a. Studio Cyclorama Background
      opaquePass.setPipeline(this.bgPipeline);
      opaquePass.draw(3, 1, 0, 0);

      // 2b. Shadow-Receiving Studio Floor
      opaquePass.setPipeline(this.floorPipeline);
      opaquePass.draw(6, 1, 0, 0);

      // 2c. Opaque Soft-Body Layers (0: Top Shell, 1: Matcha Snuffle Felt, 3: Orange Frisbee, 4: Bottom Shell) + Snuffle Treats
      opaquePass.setPipeline(this.opaqueBurgerPipeline);
      opaquePass.setVertexBuffer(0, this.vertexBuffer);
      opaquePass.setIndexBuffer(this.indexBuffer, 'uint32');

      for (const lid of [4, 3, 1, 0]) {
        if (!this.solver.isLayerActive(lid)) continue;
        const range = this.solver.layerDrawRanges[lid];
        opaquePass.drawIndexed(range.indexCount, 1, range.firstIndex, 0, 0);
      }

      if (this.solver.treatDrawRanges) {
        for (let k = 0; k < this.solver.treatDrawRanges.length; k++) {
          if (!this.solver.isTreatVisible(k)) continue;
          const trRange = this.solver.treatDrawRanges[k];
          opaquePass.drawIndexed(trRange.indexCount, 1, trRange.firstIndex, 0, 0);
        }
      }

      opaquePass.end();
    }

    // ========================================================================
    // PASS 3: Copy Rendered Opaque Scene to Refraction Background Texture
    // ========================================================================
    commandEncoder.copyTextureToTexture(
      { texture: canvasTexture },
      { texture: this.opaqueSceneTexture },
      [this.width, this.height, 1]
    );

    // ========================================================================
    // PASS 4: Refractive Translucent Pass (Layer 2 Coral-Red TPE Chew Ring)
    //         + Optional Wireframe / XPBD Lattice Overlay
    // ========================================================================
    {
      const transPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: canvasView,
          loadOp: 'load',
          storeOp: 'store'
        }],
        depthStencilAttachment: {
          view: this.depthView,
          depthLoadOp: 'load',
          depthStoreOp: 'store'
        }
      });

      transPass.setBindGroup(0, this.mainBindGroup);

      // Render Layer 2 (Coral-Red Translucent Hollow TPE Chew Ring) if active
      if (this.solver.isLayerActive(2)) {
        transPass.setPipeline(this.translucentBurgerPipeline);
        transPass.setVertexBuffer(0, this.vertexBuffer);
        transPass.setIndexBuffer(this.indexBuffer, 'uint32');
        const range2 = this.solver.layerDrawRanges[2];
        transPass.drawIndexed(range2.indexCount, 1, range2.firstIndex, 0, 0);
      }

      // Render Wireframe / XPBD Lattice / Grab Force Tether Lines
      if (lineVertCount > 0) {
        transPass.setPipeline(this.linePipeline);
        transPass.setVertexBuffer(0, this.lineVertexBuffer);
        transPass.draw(lineVertCount, 1, 0, 0);
      }

      transPass.end();
    }

    dev.queue.submit([commandEncoder.finish()]);

    return { viewProj };
  }
}
