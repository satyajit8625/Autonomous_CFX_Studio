# Autonomous CFX Studio - Third-Party Licenses, Attribution & IP Distinction

## 1. Proprietary & Original Intellectual Property (Autonomous CFX Studio)
All source code in `src/autonomous_cfx/` and associated scripts are original intellectual property created for **Autonomous CFX Studio**:
- **Autonomous CFX Agent Orchestrator**: Natural language behavior parser, diagnostic evaluator, bounded self-healing repair strategist.
- **XPBD Simulation Engine**: Vectorized Extended Position-Based Dynamics solver, distance and isometric bending constraint batches, moving collider projection with continuous contact friction.
- **Interactive CFX Studio Application**: PySide6 desktop GUI with software 3D projection viewport, timeline scrubber, agent telemetry stream, review panel.
- **Milestone Procedural Assets**: Procedural jacket, animated mannequin body, and chair colliders.
- **Interchange & Connectors**: Point cache `.vpcache` format, OBJ sequence exporter, interchange manifest, and optional Maya connector.

---

## 2. Third-Party Dependencies and Licensing

| Component | Version | License | Attribution Requirement | Redistribution Rights & Obligations |
| :--- | :--- | :--- | :--- | :--- |
| **NumPy** | `>=1.24.0` | BSD 3-Clause | Retain copyright notice and disclaimer in source distributions. | Permitted for binary and source distribution without royalty. Permissive commercial use. |
| **SciPy** | `>=1.10.0` | BSD 3-Clause | Retain copyright notice and disclaimer. | Permissive commercial use. Permitted for distribution with standard BSD attribution. |
| **Trimesh** | `>=4.0.0` | MIT License | Retain MIT copyright and permission notice. | Fully permissive. Commercial use, modification, distribution, sublicensing permitted. |
| **PySide6** (Qt for Python) | `>=6.5.0` | LGPLv3 / Commercial | Include LGPLv3 text; allow dynamic linking / library replacement. | LGPLv3 permits dynamic linking in proprietary/standalone applications without open-sourcing host application code. |
| **pytest** | `>=7.0.0` | MIT License | Test runtime only; not required for runtime deployment. | Fully permissive. |

---

## 3. Compliance and Redistribution Guidelines

1. **Standalone Operation**:
   Autonomous CFX Studio does not statically link Qt/PySide6; it loads PySide6 as dynamic Python wheels / shared libraries (`shiboken6`, Qt DLLs). Users can swap or update PySide6 binaries compliant with LGPLv3 requirements.
2. **Maya Autonomy**:
   Autodesk Maya is an **optional connector only**. The core application, solver, UI, and test suite have zero dependencies on Autodesk proprietary APIs (`maya.cmds`, `OpenMaya`). The software operates completely standalone without Maya installed.
3. **Attribution Notices**:
   In any binary or distributed package, copy this document to the top-level installation directory alongside the project `LICENSE`.
