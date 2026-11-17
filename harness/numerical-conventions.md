# Numerical conventions

These are initial working expectations, subject to project design decisions:

- Pass random keys explicitly and split them for independent random draws.
- Test representative shapes, dtypes, finite results, and known numerical
  answers. Exercise JIT, differentiation, or batching when behavior depends on
  those transformations.
- Use Equinox filtered transformations when needed for models containing both
  array and non-array leaves.
- Treat backend and precision differences as potential sources of numerical
  variation; use justified tolerances instead of assuming bitwise equality.
- Synchronize JAX results when measuring execution time and distinguish compile
  time from execution time.

Decisions still needed: default precision and x64 policy, reproducibility
requirements, expected device support, and application-specific tolerances.
Do not silently choose these on behalf of the project.
