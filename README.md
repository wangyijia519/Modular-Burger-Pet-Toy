# 🍔 Modular Burger Pet Toy — Real-Time WebGPU Multi-Material Soft-Body & Snuffle Simulator

[![Live Interactive Demo](https://img.shields.io/badge/Live_WebGPU_Demo-Click_to_Launch-E65A41?style=for-the-badge)](https://wangyijia519.github.io/Modular-Burger-Pet-Toy/)

👉 **[Click Here to Launch the Live Interactive WebGPU Demo](https://wangyijia519.github.io/Modular-Burger-Pet-Toy/)**
* **[Exploded Coaxial View (Default)](https://wangyijia519.github.io/Modular-Burger-Pet-Toy/)**
* **[Burger Snuffle Mode (All 5 Coaxial Layers Stacked)](https://wangyijia519.github.io/Modular-Burger-Pet-Toy/?mode=burger)**
* **[Snuffle Treat Dislodge Demo (Treats Popping Out of Green Felt)](https://wangyijia519.github.io/Modular-Burger-Pet-Toy/?mode=burger&snuffle=1)**
* **[Rolling Sphere Mode (Top & Bottom Shells Closed)](https://wangyijia519.github.io/Modular-Burger-Pet-Toy/?mode=sphere)**

---

## ✨ Features

* **3D Analytical Signed Distance Field (SDF) Geometry (`Ø110 mm × 112 mm`)**:
  1. **Top Shell (Bun Crown · Shore 65A Silicone)**: Cream-yellow hemispherical dome with 16 analytical rest-space SDF teardrop sesame indents and VDI-24 micro-matte grain.
  2. **Matcha-Green Wavy Ruffled Snuffle Fabric (Needle-Punched Felt)**: Double-tier organic wavy ruffled skirt (`m = 7` primary + `m = 14` secondary lobes) with Charlie cloth sheen microfacet BRDF, 3D procedural fiber noise, and **9 interactive treat-hiding ruffle pockets**.
  3. **Coral-Red Translucent Hollow TPE Chew Ring (Shore 22A Gel TPE)**: Hollow ring with 12 rounded outer dental massage bumps, screen-space refraction (`IOR = 1.44`), Beer-Lambert spectral absorption, subsurface scattering (SSS), Schlick Fresnel, and 3D embedded micro air bubbles.
  4. **Warm-Orange Frisbee Disc Module (Shore 45A Elastomer)**: Aerodynamic disc with a stepped recessed top rim well and smooth concave flight-plate dish curvature.
  5. **Bottom Shell (Notched Bowl · Shore 65A Silicone)**: Cream-yellow bowl with a front semi-circular treat/grip notch (`R = 13.5 mm`) and teardrop sesame indents.
* **Interactive Snuffle Treat Hiding & Foraging (Green Felt Layer)**:
  * Includes 9 3D SDF freeze-dried treats (**Egg Yolk Cubes**, **Marbled Salmon Bites**, and **Crunchy Star Kibble Biscuits**) tucked inside the outer ruffle pockets of the green felt layer.
  * **Pull to Dislodge**: Dragging and stretching the green felt ruffles locally opens the fold and pops hidden treats out onto the studio floor with 3D spin and bounce physics.
  * **Drag to Re-Hide**: Pick up any popped treat from the floor and drag it back onto the green felt skirt to tuck it back into a ruffle pocket.
* **Real-Time CPU XPBD Corotational Soft-Body Physics**:
  * **10 substeps @ 60 Hz** combining polar-decomposition corotational hexahedral shape-matching ($\mathbf{A}_c = \mathbf{R}_c\mathbf{S}_c$) and 5-tetrahedra-per-cell volume constraints ($\det(\mathbf{F}) = 1$).
  * Jacobian cofactor normal skinning (`updateSkinnedMesh`) for accurate lighting and refraction across large elastic deformations.

---

## 🖥️ Requirements

* **Browser**: Chrome 113+, Edge 113+, or Safari 18+ with **WebGPU** enabled.
