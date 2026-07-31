## Summary
Briefly describe what this PR does and why.

## Related issue
Closes # <!-- issue number, if applicable -->

## Type of change
- [ ] Bug fix
- [ ] New feature (new solver, assembly route, optimizer backend, utility tool)
- [ ] Improvement / refactor (no functional change)
- [ ] Documentation update
- [ ] Configuration / config.yaml change
- [ ] Other: <!-- describe -->

## Files changed
List the files changed and what was modified in each:
- `Source/`:
- `Utilities/`:
- `Examples/`:
- `config.yaml` options:

## Testing
Describe how you tested this change:

**Run tasks tested:**
- [ ] `energy` (single-point energy)
- [ ] `gradient` (single-point gradient)
- [ ] `geomopt` (geometry optimization)
- [ ] `circuits` (LUCJ circuit size analysis)

**Solvers tested:**
- [ ] FCI
- [ ] SCI
- [ ] SCI_SBD
- [ ] SQD

**Other:**
- [ ] Tested restart behavior (`restart: true`)
- [ ] Tested on HPC with Slurm
- [ ] Tested with GPU (`hf.gpu: true` or `sbd.proc_type: 1`)

Molecule / system tested on:
HPC cluster tested on (if applicable):

## Checklist
- [ ] Code follows PEP 8 style
- [ ] New functions and classes have docstrings with units and index conventions
- [ ] Mathematical operations reference relevant equations or papers in comments
- [ ] No API keys, tokens, HPC credentials, or sensitive data included
- [ ] config.yaml options are documented (inline comments or README update)
- [ ] Examples updated if new functionality is added
- [ ] CHANGELOG.md updated

## Additional notes
Any additional context for reviewers — known limitations, follow-up work needed,
or areas to pay close attention to during review.
