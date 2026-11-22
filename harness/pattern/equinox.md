# Equinox abstract/final pattern

Use the [Equinox abstract/final pattern](https://docs.kidger.site/equinox/pattern/)
when designing module hierarchies. It makes interfaces, implementations, and
initialization easier to follow. These are design conventions; do not assume
that Equinox enforces every rule automatically.

- Each `eqx.Module` subclass is either abstract (subclassable, but not
  instantiable) or final (instantiable, but not subclassable). Name abstract
  classes with an `Abstract` prefix.
- Declare interfaces with `abc.abstractmethod` and required instance
  attributes with `eqx.AbstractVar`. Use `eqx.AbstractClassVar` for required
  class attributes. Declare requirements explicitly rather than probing with
  `hasattr`.
- Implement an abstract method or attribute once. Do not override a concrete
  implementation, even in an intermediate abstract class. An abstract class
  may provide shared implementations while leaving other requirements abstract.
- Keep all concrete dataclass fields and initialization in one class, usually
  the final class. Do not split initialization across a hierarchy or use
  `super()` to chain it.
- To adapt a concrete module, compose or wrap it instead of subclassing it.
  Multiple inheritance of abstract interfaces is compatible with this pattern.
- Use `__check_init__` for intrinsic invariants that should be checked after
  initialization, including invariants declared by abstract classes. Checks
  validate state; they should not mutate fields.

For example, an intermediate abstract class can provide behavior while the
final class owns its fields:

```python
import abc

import equinox as eqx
import jax


class AbstractTransform(eqx.Module):
    @abc.abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError


class AbstractScaledTransform(AbstractTransform):
    scale: eqx.AbstractVar[float]

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.scale * x

    def __check_init__(self):
        if self.scale <= 0:
            raise ValueError("scale must be positive")


class Scaling(AbstractScaledTransform):
    scale: float
```

`AbstractScaledTransform` remains abstract until `scale` is supplied. `Scaling`
uses Equinox's generated initializer and is treated as final; its inherited
check validates the scalar configuration after construction.
