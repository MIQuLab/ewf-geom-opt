# Contributing to ewf-geom-opt

Thank you for your interest in contributing to this project. This repository is developed and maintained by [MIQuLab](https://github.com/MIQuLab) and welcomes contributions from collaborators and the broader research community.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Contact](#contact)

---

## Getting Started

1. **Fork** the repository to your own GitHub account
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/ewf-geom-opt.git
   cd ewf-geom-opt
   ```
3. **Install dependencies** — see the Requirements section in README.md
4. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. Make your changes, test them, then push and open a Pull Request

---

## Reporting Bugs

If you find a bug, please open a [Bug Report](https://github.com/MIQuLab/ewf-geom-opt/issues/new?template=bug_report.yml)
using our structured issue form. The form will guide you through providing
all the information we need — run task, solver, environment versions,
config.yaml, and error output.

> **Note:** Remove any API keys, HPC credentials, or sensitive account
> details from your config before submitting.

---

## Suggesting Enhancements

For new features or improvements, open a
[Feature Request](https://github.com/MIQuLab/ewf-geom-opt/issues/new?template=feature_request.yml)
using our structured issue form. The form will guide you through describing
the motivation, proposed approach, and relevant references.

> **Note:** For larger changes (new density-assembly routes, new solver
> integrations, new optimizer backends), please open an issue to discuss
> the approach **before** starting implementation — this avoids duplicated
> effort and ensures alignment with the project's direction.

---

## Submitting Changes

1. **Keep commits focused** — one logical change per commit
2. **Write clear commit messages**:
   ```
   Short summary of the change (50 characters or less)

   Longer explanation of what changed and why, if needed.
   Reference any related issue: Fixes #42
   ```
3. **Test your changes** before submitting:
   - Run an example from `Examples/` to verify nothing is broken
   - If adding a new assembly route or solver, include a minimal test case
4. **Open a Pull Request** against the `main` branch:
   - Describe what the PR does and why
   - Reference any related issues
   - At least one review from a MIQuLab maintainer is required before merging

---

## Code Style

- Follow standard **PEP 8** Python style
- Use descriptive variable names — this is a scientific codebase and clarity matters
- Add docstrings to new functions and classes, especially:
  - Mathematical description of what the function computes
  - Units and index conventions (MO vs AO basis, occupied/virtual ordering)
  - References to equations in papers where applicable
- Keep new solver modules consistent with the structure of existing ones
  (`sqd_solver.py`, `external_sci.py`) so the driver dispatch remains clean

---

## Contact

For questions about the science, methodology, or collaboration opportunities:

**MIQuLab Team**
📧 miqulab.team@gmail.com
🌐 https://github.com/MIQuLab

For questions specifically about the SBD eigensolver binary, refer to the
[SBD repository](https://github.com/r-ccs-cms/sbd).

For questions about the sister project and Qiskit integration, refer to
[quantum-fragment-methods](https://github.com/qiskit-community/quantum-fragment-methods).
