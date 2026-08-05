# Documentation

Topic-by-topic documentation for the EWF-based geometry-optimization workflow.
See the [project README](../README.md) for an overview.

| Document | Contents |
|---|---|
| [Installation](Installation.md) | Core and optional dependencies — optimizer backends, GPU-accelerated HF, the external SBD eigensolver, SQD quantum sampling |
| [Theory](Theory.md) | The EWF energy, its analytic nuclear gradient, and the density-response term |
| [Density-assembly routes](Density_Assembly_Routes.md) | The routes that turn per-fragment solutions into one global density, and how they differ |
| [Run modes and tasks](Run_Modes_and_Tasks.md) | What `run_mode` and `run_task` select |
| [Usage](Usage.md) | Generating a `config.yaml`, the full option reference, solver and Slurm settings |
| [Running](Running.md) | Launching the workflow, worker modes, and restarting an interrupted run |

[`Images/`](Images/) holds every figure referenced from any README in this
repository.
