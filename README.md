# Autonomous CFX Studio

> **Next-generation standalone CFX simulation and autonomous cloth agent.**

Autonomous CFX Studio is an independent, production-grade application engineered for character effects artists and automated CFX pipelines. It provides a real-time/interactive XPBD (Extended Position-Based Dynamics) physical cloth solver, paired with an autonomous agent orchestrator that parses natural-language behavior descriptions, executes simulations, evaluates quality against physical and production metrics, applies bounded self-healing repairs, and exports approved caches.

---

## Key Capabilities

1. **Autonomous Workflow Milestone**:
   - **Import Animated Geometry**: Ingests cloth meshes and animated collider mannequins/obstacles.
   - **Describe Cloth Behavior**: Interprets artist descriptions (e.g. *"heavy leather jacket, stiff collar, low stretch, high friction with body, high damping"*).
   - **Agent Prepares and Runs Simulation**: Automatically sets up XPBD compliance, solver substeps, collision thickness, and pre-relaxes initial overlaps.
   - **Automated Quality Evaluation**: Diagnostic checks for penetration depth, edge stretch strain, velocity spikes, and numerical stability.
   - **Bounded Self-Healing Repairs**: Automatically applies targeted diagnostic adjustments (adaptive substeps, expanded collision margins, compliance tightening) within bounded iteration limits.
   - **Artist Review & Approval**: Interactive 3D review with timeline scrubbing, quality score badge, and pass/fail inspection.
   - **Export Approved Caches**: Exports point caches (`.vpcache`), Wavefront OBJ sequences, and standard interchange manifests (Alembic/USD readiness).

2. **Standalone & Independent**:
   - Runs 100% standalone without Autodesk Maya installed.
   - Maya support is an optional connector for pipeline interoperability.
   - Built on a fresh architecture with clean separation of concerns.

3. **Production Assets Included**:
   - **Procedural Jacket**: Torso, sleeves, collar, front slit, vertex groups.
   - **Animated Body Collider**: Mannequin with procedural walking, torso twist, and sitting motion.
   - **Chair Collider**: Seat, backrest, and legs collider.

---

## Quick Start (Windows)

### 1. Installation
Run the installer script:
```cmd
scripts\install.bat
```
Or manually via pip:
```cmd
pip install -r requirements.txt
```

### 2. Launch Interactive Desktop GUI
```cmd
scripts\run_studio.bat
```

### 3. Headless Pipeline Execution
```cmd
python scripts\run_headless.py --prompt "heavy leather jacket, stiff collar, low stretch" --frames 30
```

### 4. Run Test Suite
```cmd
python -m pytest tests -v
```

---

## Project Structure

```text
Cfx_Studio/
├── src/
│   └── autonomous_cfx/
│       ├── core/          # Mesh geometry, Point cache I/O, Project hierarchy, Logger
│       ├── simulation/    # XPBD solver, Constraints, Animated colliders, Engine
│       ├── agent/         # Behavior parser, Evaluator, Bounded repair strategist, Orchestrator
│       ├── assets/        # Procedural Jacket, Body, Chair generators, Asset manager
│       ├── jobs/          # Threaded background job runner
│       ├── connectors/    # Optional Maya connector
│       └── ui/            # PySide6 desktop application, Viewport, Timeline, Panels
├── scripts/
│   ├── install.bat        # Windows dependency installation
│   ├── run_studio.bat     # Studio GUI launcher
│   └── run_headless.py    # Headless pipeline batch runner
├── tests/                 # Automated Pytest suite
├── THIRD_PARTY_LICENSES.md
└── Autonomous_CFX_Studio_Review.md
```
