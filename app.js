// ============================================================================
// app.js
// Main Application Controller, Interactive 3D Raycast Grabbing, Orbit Camera,
// Smooth Mode Transitions, 3D Callout Projection & Live Telemetry UI
// ============================================================================

import { LAYER_CONFIGS, generateLayerMesh } from './sdf_mesh.js';
import { BurgerXPBDSolver } from './xpbd_physics.js';
import { BurgerWebGPURenderer, mat4Invert } from './webgpu_renderer.js';

class ModularBurgerApp {
  constructor() {
    this.canvas = document.getElementById('webgpu-canvas');
    this.calloutContainer = document.getElementById('callout-overlay');
    this.axisLineEl = document.getElementById('coaxial-axis-line');

    // Orbit camera state matching the reference photo's clean ¾ studio perspective
    this.camera = {
      theta: 0.14,       // azimuth angle (radians)
      phi: 1.22,         // polar angle from +Y (radians, ~20 deg elevation)
      distance: 6.45,    // camera distance to frame full exploded stack comfortably
      target: [0.0, 0.02, 0.0],
      eye: [0.0, 0.8, 6.45]
    };

    this.lastViewProj = null;
    this.isOrbiting = false;
    this.lastPointer = [0, 0];
    this.showCallouts = true;

    // Smooth animated explosion gap target
    this.targetGapMm = 26.0;
    this.animatingGap = false;

    this.fpsSmooth = 60.0;
    this.lastFrameTime = performance.now();
  }

  async start() {
    const statusEl = document.getElementById('loading-status');
    try {
      statusEl.textContent = 'Generating 5 Coaxial Layer Meshes from Rest-Space 3D SDFs...';

      // 1. Extract high-resolution surface meshes from analytical 3D SDFs
      const t0 = performance.now();
      this.layerMeshes = LAYER_CONFIGS.map(cfg => generateLayerMesh(cfg));
      const sdfTimeMs = (performance.now() - t0).toFixed(1);

      statusEl.textContent = 'Building Hexahedral/Tetrahedral XPBD Lattice & Trilinear Skinning...';

      // 2. Initialize CPU XPBD Corotational Shape-Matching & Tetrahedral Volume Solver
      this.solver = new BurgerXPBDSolver(this.layerMeshes);

      statusEl.textContent = 'Initializing WebGPU Multi-Pass Refraction & Shadow Pipeline...';

      // 3. Initialize WebGPU Renderer
      this.renderer = new BurgerWebGPURenderer(this.canvas, this.solver);
      await this.renderer.init();

      // Hide loading overlay immediately
      const loader = document.getElementById('loading-overlay');
      if (loader) {
        loader.remove();
      }

      const meshStatsEl = document.getElementById('stat-mesh-info');
      if (meshStatsEl) {
        meshStatsEl.textContent =
          `${(this.solver.totalRenderVerts / 1000).toFixed(1)}k verts · ${this.solver.hexCells.length} hex / ${this.solver.tets.length} tets (${sdfTimeMs}ms SDF)`;
      }

      this._bindUIControls();
      this._bindPointerInteractions();
      this._buildCalloutDOM();

      // Optional URL query parameter support for mode / mesh inspection
      const params = new URLSearchParams(window.location.search);
      const qMode = params.get('mode');
      if (qMode === 'burger') {
        document.getElementById('mode-burger').click();
        this.solver.explosionGapMm = 0.0;
        this.animatingGap = false;
        const gs = document.getElementById('slider-gap');
        const gv = document.getElementById('val-gap');
        if (gs) gs.value = '0';
        if (gv) gv.textContent = '0 mm';
        this.camera.distance = 4.95;
        this.camera.target = [0.0, -0.05, 0.0];
        this.solver.resetSimulation(true);
      } else if (qMode === 'sphere') {
        document.getElementById('mode-sphere').click();
        this.camera.distance = 4.65;
        this.camera.target = [0.0, -0.05, 0.0];
        this.solver.resetSimulation(true);
      }
      if (params.get('mesh') === '1') {
        document.getElementById('btn-mesh').click();
      }

      window.addEventListener('resize', () => {
        this.renderer.resize(this.canvas.clientWidth, this.canvas.clientHeight);
      });

      if (params.get('snuffle') === '1') {
        setTimeout(() => {
          this.solver.snuffleDislodgeTreats(4);
        }, 300);
      } else if (!params.has('nopoke')) {
        setTimeout(() => {
          this.solver.triggerPoke();
        }, 350);
      }

      this._frameLoop(performance.now());
    } catch (err) {
      console.error(err);
      const loader = document.getElementById('loading-overlay');
      if (loader) {
        loader.innerHTML = `
          <div class="error-box">
            <h3>WebGPU Initialization Error</h3>
            <p>${err.message}</p>
            <p class="error-hint">Please open this page in Chrome 113+, Edge 113+, or Safari 18+ with WebGPU enabled.</p>
          </div>
        `;
      }
    }
  }

  _updateCameraEye() {
    const sinP = Math.sin(this.camera.phi);
    const cosP = Math.cos(this.camera.phi);
    const sinT = Math.sin(this.camera.theta);
    const cosT = Math.cos(this.camera.theta);

    this.camera.eye[0] = this.camera.target[0] + this.camera.distance * sinP * sinT;
    this.camera.eye[1] = this.camera.target[1] + this.camera.distance * cosP;
    this.camera.eye[2] = this.camera.target[2] + this.camera.distance * sinP * cosT;
  }

  _screenToWorldRay(clientX, clientY) {
    if (!this.lastViewProj) return null;
    const rect = this.canvas.getBoundingClientRect();
    const nx = ((clientX - rect.left) / rect.width) * 2.0 - 1.0;
    const ny = 1.0 - ((clientY - rect.top) / rect.height) * 2.0;

    const invVP = mat4Invert(this.lastViewProj);

    const unproject = (zNdc) => {
      const x = invVP[0] * nx + invVP[4] * ny + invVP[8] * zNdc + invVP[12];
      const y = invVP[1] * nx + invVP[5] * ny + invVP[9] * zNdc + invVP[13];
      const z = invVP[2] * nx + invVP[6] * ny + invVP[10] * zNdc + invVP[14];
      const w = invVP[3] * nx + invVP[7] * ny + invVP[11] * zNdc + invVP[15];
      const invW = 1.0 / (w || 1e-9);
      return [x * invW, y * invW, z * invW];
    };

    const pNear = unproject(0.0);
    const pFar = unproject(1.0);
    let dx = pFar[0] - pNear[0];
    let dy = pFar[1] - pNear[1];
    let dz = pFar[2] - pNear[2];
    const len = Math.hypot(dx, dy, dz) + 1e-12;
    return {
      origin: this.camera.eye.slice(),
      dir: [dx / len, dy / len, dz / len]
    };
  }

  _bindPointerInteractions() {
    const c = this.canvas;

    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      this.lastPointer = [e.clientX, e.clientY];

      // Left-click / primary touch tries 3D soft-body raycast grab or treat hiding first
      if (e.button === 0 && !e.altKey) {
        const ray = this._screenToWorldRay(e.clientX, e.clientY);
        if (ray && this.solver.beginGrab(ray.origin, ray.dir, e.shiftKey)) {
          if (this.solver.grabState.active) {
            c.style.cursor = 'grabbing';
            this._highlightLayerCard(this.solver.grabState.layerId);
          }
          return;
        }
      }

      this.isOrbiting = true;
      c.style.cursor = 'move';
    });

    c.addEventListener('pointermove', (e) => {
      const dx = e.clientX - this.lastPointer[0];
      const dy = e.clientY - this.lastPointer[1];
      this.lastPointer = [e.clientX, e.clientY];

      if (this.solver.grabState.active) {
        const ray = this._screenToWorldRay(e.clientX, e.clientY);
        if (ray) {
          this.solver.moveGrab(ray.origin, ray.dir);
        }
        return;
      }

      if (this.isOrbiting) {
        this.camera.theta -= dx * 0.0075;
        this.camera.phi = Math.max(0.28, Math.min(2.65, this.camera.phi - dy * 0.0065));
      }
    });

    const releasePointer = () => {
      if (this.solver.grabState.active) {
        this.solver.endGrab();
        this._highlightLayerCard(-1);
      }
      this.isOrbiting = false;
      c.style.cursor = 'grab';
    };

    c.addEventListener('pointerup', releasePointer);
    c.addEventListener('pointercancel', releasePointer);

    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.camera.distance = Math.max(2.35, Math.min(8.8, this.camera.distance + e.deltaY * 0.0028));
    }, { passive: false });
  }

  _bindUIControls() {
    // Mode switcher buttons
    const modeBtns = {
      exploded: document.getElementById('mode-exploded'),
      burger: document.getElementById('mode-burger'),
      sphere: document.getElementById('mode-sphere')
    };

    const gapSlider = document.getElementById('slider-gap');
    const gapValEl = document.getElementById('val-gap');
    const dimBadgeEl = document.getElementById('active-dim-badge');

    const setMode = (newMode) => {
      this.solver.mode = newMode;
      Object.keys(modeBtns).forEach(k => {
        modeBtns[k].classList.toggle('active', k === newMode);
      });

      if (newMode === 'exploded') {
        this.targetGapMm = Math.max(18.0, this.solver.explosionGapMm || 26.0);
        this.animatingGap = true;
        dimBadgeEl.textContent = `Exploded Coaxial Stack · Ø110 mm`;
      } else if (newMode === 'burger') {
        this.targetGapMm = 0.0;
        this.animatingGap = true;
        dimBadgeEl.textContent = `Burger Snuffle Mode · Ø110 mm × 112 mm`;
      } else if (newMode === 'sphere') {
        this.targetGapMm = 0.0;
        this.solver.explosionGapMm = 0.0;
        this.animatingGap = false;
        gapSlider.value = '0';
        gapValEl.textContent = '0 mm';
        dimBadgeEl.textContent = `Rolling Sphere Mode · Ø110 mm Closed Ball`;
        this.solver.resetSimulation(true);
        this.solver.triggerPoke();
      }
    };

    modeBtns.exploded.addEventListener('click', () => setMode('exploded'));
    modeBtns.burger.addEventListener('click', () => setMode('burger'));
    modeBtns.sphere.addEventListener('click', () => setMode('sphere'));

    // Stiffness slider
    const stiffSlider = document.getElementById('slider-stiffness');
    const stiffValEl = document.getElementById('val-stiffness');
    stiffSlider.addEventListener('input', () => {
      const v = parseFloat(stiffSlider.value);
      this.solver.stiffnessScale = v;
      stiffValEl.textContent = `${v.toFixed(2)}×`;
    });

    // Internal Damping slider
    const dampSlider = document.getElementById('slider-damping');
    const dampValEl = document.getElementById('val-damping');
    dampSlider.addEventListener('input', () => {
      const v = parseFloat(dampSlider.value);
      this.solver.dampingScale = v;
      dampValEl.textContent = v.toFixed(2);
    });

    // Explosion Gap slider
    gapSlider.addEventListener('input', () => {
      const v = parseFloat(gapSlider.value);
      this.animatingGap = false;
      this.solver.explosionGapMm = v;
      gapValEl.textContent = `${Math.round(v)} mm`;

      if (this.solver.mode === 'sphere' && v > 1.0) {
        this.solver.mode = 'exploded';
        Object.keys(modeBtns).forEach(k => modeBtns[k].classList.toggle('active', k === 'exploded'));
        dimBadgeEl.textContent = `Exploded Coaxial Stack · Ø110 mm`;
      } else if (this.solver.mode === 'burger' && v > 2.0) {
        this.solver.mode = 'exploded';
        Object.keys(modeBtns).forEach(k => modeBtns[k].classList.toggle('active', k === 'exploded'));
        dimBadgeEl.textContent = `Exploded Coaxial Stack · Ø110 mm`;
      } else if (this.solver.mode === 'exploded' && v <= 0.5) {
        this.solver.mode = 'burger';
        Object.keys(modeBtns).forEach(k => modeBtns[k].classList.toggle('active', k === 'burger'));
        dimBadgeEl.textContent = `Burger Snuffle Mode · Ø110 mm × 112 mm`;
      }
    });

    // Action Buttons: Poke, Reset, 1/4 Speed, Show Mesh, Pause, Callouts
    document.getElementById('btn-poke').addEventListener('click', () => {
      this.solver.triggerPoke();
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
      this.solver.resetSimulation(true);
    });

    const btnSlow = document.getElementById('btn-slow');
    btnSlow.addEventListener('click', () => {
      this.solver.slowMo = !this.solver.slowMo;
      btnSlow.classList.toggle('active', this.solver.slowMo);
    });

    const btnMesh = document.getElementById('btn-mesh');
    btnMesh.addEventListener('click', () => {
      this.renderer.showMesh = !this.renderer.showMesh;
      btnMesh.classList.toggle('active', this.renderer.showMesh);
    });

    const btnPause = document.getElementById('btn-pause');
    btnPause.addEventListener('click', () => {
      this.solver.paused = !this.solver.paused;
      btnPause.classList.toggle('active', this.solver.paused);
      btnPause.textContent = this.solver.paused ? 'Resume' : 'Pause';
    });

    const btnCallouts = document.getElementById('btn-callouts');
    if (btnCallouts) {
      btnCallouts.addEventListener('click', () => {
        this.showCallouts = !this.showCallouts;
        btnCallouts.classList.toggle('active', this.showCallouts);
      });
    }

    // Snuffle Treat Foraging Controls (Layer 1 Matcha Green Felt)
    const btnStuffTreats = document.getElementById('btn-stuff-treats');
    if (btnStuffTreats) {
      btnStuffTreats.addEventListener('click', () => {
        this.solver.stuffAllTreats(true);
      });
    }

    const btnSnufflePop = document.getElementById('btn-snuffle-pop');
    if (btnSnufflePop) {
      btnSnufflePop.addEventListener('click', () => {
        this.solver.snuffleDislodgeTreats(3);
      });
    }

    const btnClickFeltMode = document.getElementById('btn-click-felt-mode');
    if (btnClickFeltMode) {
      btnClickFeltMode.addEventListener('click', () => {
        this.solver.clickFeltToHide = !this.solver.clickFeltToHide;
        btnClickFeltMode.classList.toggle('active', this.solver.clickFeltToHide);
      });
    }

    const btnClearTreats = document.getElementById('btn-clear-treats');
    if (btnClearTreats) {
      btnClearTreats.addEventListener('click', () => {
        this.solver.clearAllTreats();
      });
    }

    // Mobile drawer toggle
    const mobileToggle = document.getElementById('mobile-panel-toggle');
    const controlSidebar = document.getElementById('control-sidebar');
    if (mobileToggle && controlSidebar) {
      mobileToggle.addEventListener('click', () => {
        controlSidebar.classList.toggle('collapsed');
      });
    }
  }

  _buildCalloutDOM() {
    this.calloutContainer.innerHTML = '';
    this.calloutEls = [];

    // Local Y attachment offsets for the 5 layers
    this.calloutLocalPos = [
      [0.86, 0.36, 0.0],  // Layer 0: Top Shell
      [0.92, 0.00, 0.0],  // Layer 1: Matcha Snuffle Fabric
      [0.92, 0.00, 0.0],  // Layer 2: Coral-Red TPE Chew Ring
      [0.86, 0.00, 0.0],  // Layer 3: Warm-Orange Frisbee Disc
      [0.88, -0.36, 0.0]  // Layer 4: Bottom Shell
    ];

    for (let lid = 0; lid < 5; lid++) {
      const cfg = LAYER_CONFIGS[lid];
      const item = document.createElement('div');
      item.className = 'layer-callout';
      item.innerHTML = `
        <span class="callout-dot"></span>
        <span class="callout-line"></span>
        <div class="callout-card" data-layer="${lid}">
          <span class="callout-swatch" style="background:${cfg.colorHex}"></span>
          <div class="callout-text">
            <strong>${cfg.name}</strong>
            <small>${cfg.materialType}</small>
          </div>
        </div>
      `;
      item.querySelector('.callout-card').addEventListener('click', () => {
        this.solver.triggerPoke();
      });
      this.calloutContainer.appendChild(item);
      this.calloutEls.push(item);
    }
  }

  _updateCalloutPositions() {
    if (!this.lastViewProj || !this.calloutEls) return;

    const show = this.showCallouts && this.solver.mode === 'exploded' && this.solver.explosionGapMm > 10.0;
    this.calloutContainer.style.opacity = show ? '1' : '0';
    this.calloutContainer.style.pointerEvents = show ? 'auto' : 'none';
    if (this.axisLineEl) {
      this.axisLineEl.style.opacity = show ? '1' : '0';
    }
    if (!show) return;

    const rect = this.canvas.getBoundingClientRect();
    const vp = this.lastViewProj;

    for (let lid = 0; lid < 5; lid++) {
      const el = this.calloutEls[lid];
      const localP = this.calloutLocalPos[lid];
      const wx = localP[0];
      const wy = localP[1] + this.solver.getLayerTargetY(lid);
      const wz = localP[2];

      const cx = vp[0] * wx + vp[4] * wy + vp[8] * wz + vp[12];
      const cy = vp[1] * wx + vp[5] * wy + vp[9] * wz + vp[13];
      const cw = vp[3] * wx + vp[7] * wy + vp[11] * wz + vp[15];

      if (cw <= 0.1) {
        el.style.display = 'none';
        continue;
      }
      el.style.display = 'flex';
      const sx = ((cx / cw) * 0.5 + 0.5) * rect.width;
      const sy = (0.5 - (cy / cw) * 0.5) * rect.height;

      el.style.transform = `translate3d(${sx.toFixed(1)}px, ${sy.toFixed(1)}px, 0)`;
    }
  }

  _highlightLayerCard(layerId) {
    const cards = document.querySelectorAll('.cmf-layer-row');
    cards.forEach((c) => {
      const id = parseInt(c.getAttribute('data-layer-id'), 10);
      c.classList.toggle('active-grab', id === layerId);
    });
  }

  _frameLoop(nowMs) {
    const dtMs = Math.max(1.0, nowMs - this.lastFrameTime);
    this.lastFrameTime = nowMs;
    this.fpsSmooth = this.fpsSmooth * 0.92 + (1000.0 / dtMs) * 0.08;

    // Smoothly animate explosionGapMm if mode transition is in progress
    if (this.animatingGap) {
      const diff = this.targetGapMm - this.solver.explosionGapMm;
      if (Math.abs(diff) < 0.15) {
        this.solver.explosionGapMm = this.targetGapMm;
        this.animatingGap = false;
      } else {
        this.solver.explosionGapMm += diff * 0.14;
      }
      const gapSlider = document.getElementById('slider-gap');
      const gapValEl = document.getElementById('val-gap');
      if (gapSlider && gapValEl) {
        gapSlider.value = this.solver.explosionGapMm.toFixed(1);
        gapValEl.textContent = `${Math.round(this.solver.explosionGapMm)} mm`;
      }
    }

    // 1. Step CPU XPBD Corotational Shape-Matching & Tetrahedral Volume Simulation (10 substeps @ 60 Hz)
    const telemetry = this.solver.step();

    // 2. Render WebGPU Multi-Pass Frame
    this._updateCameraEye();
    const renderInfo = this.renderer.render(this.camera, nowMs * 0.001);
    if (renderInfo && renderInfo.viewProj) {
      this.lastViewProj = renderInfo.viewProj;
    }

    // 3. Update 3D Callout Labels & Live Telemetry Readouts
    this._updateCalloutPositions();
    this._updateTelemetryDOM(telemetry);

    requestAnimationFrame((t) => this._frameLoop(t));
  }

  _updateTelemetryDOM(telemetry) {
    const massEl = document.getElementById('readout-mass');
    const massSubEl = document.getElementById('readout-mass-sub');
    const volEl = document.getElementById('readout-volume');
    const volRatioEl = document.getElementById('readout-vol-ratio');
    const keEl = document.getElementById('readout-ke');
    const keBarEl = document.getElementById('ke-meter-fill');
    const fpsEl = document.getElementById('readout-fps');
    const grabBadgeEl = document.getElementById('grab-telemetry-badge');
    const treatStatusEl = document.getElementById('treat-status-badge');

    if (massEl) massEl.textContent = `${telemetry.massG.toFixed(1)} g`;
    if (massSubEl) {
      massSubEl.textContent = telemetry.hiddenTreatsCount > 0
        ? `+${telemetry.treatsMassG.toFixed(1)}g (${telemetry.hiddenTreatsCount} treats)`
        : `Coaxial Stack`;
    }
    if (volEl) volEl.textContent = `${telemetry.volumeCm3.toFixed(1)} cm³`;
    if (volRatioEl) {
      volRatioEl.textContent = `${telemetry.volRatioPct.toFixed(2)}% V₀`;
    }
    if (keEl) keEl.textContent = `${telemetry.kineticEnergyMJ.toFixed(2)} mJ`;
    if (keBarEl) {
      const pct = Math.min(100.0, Math.sqrt(telemetry.kineticEnergyMJ) * 18.0);
      keBarEl.style.width = `${pct.toFixed(1)}%`;
    }
    if (fpsEl) {
      fpsEl.textContent = `${Math.round(this.fpsSmooth)} FPS · 10 Substeps`;
    }
    if (treatStatusEl) {
      treatStatusEl.textContent = `${telemetry.hiddenTreatsCount} Hidden · ${telemetry.freeTreatsCount} Popped`;
    }
    if (grabBadgeEl) {
      if (this.solver.grabState.active) {
        if (this.solver.grabState.treatIdx !== -1) {
          grabBadgeEl.style.display = 'inline-flex';
          grabBadgeEl.textContent = `Dragging Treat #${this.solver.grabState.treatIdx + 1} (Drop on Green Felt to Hide)`;
        } else {
          const lid = this.solver.grabState.layerId;
          const lName = LAYER_CONFIGS[lid] ? LAYER_CONFIGS[lid].shortName : 'Layer';
          grabBadgeEl.style.display = 'inline-flex';
          grabBadgeEl.textContent = `Pulling ${lName} · ${telemetry.grabForceN.toFixed(1)} N`;
        }
      } else {
        grabBadgeEl.style.display = 'none';
      }
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const app = new ModularBurgerApp();
  app.start();
});
