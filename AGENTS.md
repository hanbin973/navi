# Agent instructions

This repository is starting a new JAX/Equinox project. The harness is a draft:
confirm unresolved project and cluster settings before relying on them.

## Route by task

- Environment, dependencies, or uv: read `harness/environment.md`.
- Editing and verification: read `harness/development.md`.
- JAX/Equinox behavior or numerical tests: read `harness/numerical-conventions.md`.
- Slurm or compute resources: read `harness/cluster.md` before running jobs.

Use available JAX/Equinox skills when relevant. Project and site instructions
take precedence over generic examples. Do not infer unrecorded cluster settings.

## Working rules

- Preserve user changes and keep work within the requested scope.
- Use the project uv environment; do not install project dependencies with pip.
- Keep substantial computation off the login node. Follow site policies for
  filesystem access and scheduler polling.
- Run checks relevant to the change and report what passed or could not run.
- Update the corresponding harness file when a project convention changes.
