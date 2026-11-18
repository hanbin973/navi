import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from navi import Navi


def make_model(**overrides):
    settings = {
        "context_size": 3,
        "query_size": 2,
        "pair_size": 2,
        "out_size": 2,
        "num_ids": 6,
        "width": 4,
        "num_heads": 2,
        "query_depth": 3,
        "zero_init": False,
        "key": jax.random.key(10),
    }
    return Navi(**(settings | overrides))


@pytest.fixture(scope="module")
def episode():
    x, u, pairs = jax.random.split(jax.random.key(11), 3)
    return (
        jax.random.normal(x, (3, 3)),
        jnp.array([4, 1, 3]),
        jax.random.normal(u, (4, 2)),
        jax.random.normal(pairs, (4, 3, 2)),
    )


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "override",
    [
        {"query_depth": 0},
        {"num_heads": 0},
        {"num_heads": 3},
        {"num_ids": -1},
        {"context_size": 0},
        {"query_size": 0},
        {"pair_size": 0},
        {"out_size": 0},
        {"width": 0},
        {"hidden_size": 0},
        {"query_depth": 1.5},
    ],
)
def test_constructor_validation(override):
    with pytest.raises(ValueError):
        make_model(**override)


def test_input_validation(episode):
    model = make_model()
    x, ids, u, pairs = episode
    for bad_x in (x[0], x[:0], x[:, :2]):
        with pytest.raises(ValueError, match="x must"):
            model.encode_context(bad_x, ids)
    for bad_ids in (
        ids[:, None],
        ids.at[0].set(-1),
        ids.at[0].set(6),
        np.array([2**32 + 4, 1, 3], dtype=np.uint64),
    ):
        with pytest.raises(ValueError, match="ids must"):
            model(x, bad_ids, u, pairs)
    with pytest.raises(TypeError, match="ids must"):
        model(x, ids.astype(jnp.float32), u, pairs)
    with pytest.raises(TypeError, match="x must"):
        model(x.astype(jnp.int32), ids, u, pairs)
    context = model.encode_context(x, ids)
    with pytest.raises(ValueError, match="one state"):
        model.decode_queries(context[1:], u, pairs)
    with pytest.raises(ValueError, match="context state"):
        model.decode_queries((*context[:-1], context[-1][:-1]), u, pairs)
    with pytest.raises(ValueError, match="u must"):
        model.decode_queries(context, u[0], pairs)
    with pytest.raises(ValueError, match="pair_features must"):
        model.decode_queries(context, u, pairs[:, :-1])
    with pytest.raises(TypeError, match="pair_features must"):
        model.decode_queries(context, u, pairs.astype(jnp.int32))


def test_one_layer_and_custom_head(episode):
    model = make_model(query_depth=1, head=lambda token: token[:2])
    x, ids, u, pairs = episode
    context = model.encode_context(x, ids)
    assert len(context) == 1
    assert not model.context_blocks
    close(
        context[0],
        jax.vmap(model.context_projection)(x) + model.id_embedding.weight[ids],
    )
    assert model(x, ids, u, pairs).shape == (4, 2)


def test_traced_id_validation(episode):
    model = make_model(query_depth=1, num_ids=130)
    x, ids, u, pairs = episode
    run = eqx.filter_jit(lambda model, ids: model(x, ids, u, pairs))
    close(run(model, ids.astype(jnp.int8)), model(x, ids, u, pairs))
    with pytest.raises(eqx.EquinoxRuntimeError, match=r"ids must be in"):
        run(model, ids.at[0].set(130)).block_until_ready()


def test_cached_and_permuted_reads(episode):
    model = make_model()
    x, ids, u, pairs = episode
    context = model.encode_context(x, ids)
    expected = model(x, ids, u, pairs)
    assert expected.shape == (4, 2)
    assert expected.dtype == jnp.float32
    read = model.decode_queries
    close(read(context, u, pairs), expected)
    close(read(context, u[2:3], pairs[2:3]), expected[2:3])
    close(
        jnp.concatenate(
            [read(context, u[:1], pairs[:1]), read(context, u[1:], pairs[1:])]
        ),
        expected,
    )
    assert read(context, u[:0], pairs[:0]).shape == (0, 2)
    queries = jnp.array([2, 0, 3, 1])
    close(read(context, u[queries], pairs[queries]), expected[queries])
    rows = jnp.array([1, 2, 0])
    close(model(x[rows], ids[rows], u, pairs[:, rows]), expected)
    for before, after in zip(context, model.encode_context(x, ids), strict=True):
        np.testing.assert_array_equal(before, after)


def manual_read(model, context, query, pairs):
    token = model.query_projection(query)[None]
    for layer in range(len(model.query_blocks)):
        encoder = model.pair_encoders[layer]
        memory = jnp.stack(
            [
                encoder(jnp.concatenate((context[layer][row], pairs[row])))
                for row in range(len(pairs))
            ]
        )
        token = model.query_blocks[layer](token, memory)
    return model.head(token[0])


def test_layer_schedule(episode):
    model = make_model()
    x, ids, u, pairs = episode
    states = model.encode_context(x, ids)
    before = jax.vmap(model.context_projection)(x) + model.id_embedding.weight[ids]
    close(states[0], before)
    for state, block in zip(states[1:], model.context_blocks, strict=True):
        before = block(before)
        close(state, before)

    distinguishable = tuple(
        jax.random.normal(jax.random.key(layer), states[0].shape) for layer in range(3)
    )
    actual = model.decode_queries(distinguishable, u[:1], pairs[:1])[0]
    close(actual, manual_read(model, distinguishable, u[0], pairs[0]))
    swapped = manual_read(model, distinguishable[::-1], u[0], pairs[0])
    assert not np.allclose(actual, swapped)


def test_batched_training_matches_independent_loop(episode):
    model = make_model()
    batch = tuple(jnp.stack([value, value[::-1]]) for value in episode)
    target = jnp.linspace(-0.3, 0.7, 16).reshape(2, 4, 2)

    def batched(model, batch):
        return jax.vmap(model)(*batch)

    def loop(model, batch):
        outputs = []
        for x, ids, u, pairs in zip(*batch, strict=True):
            context = model.encode_context(x, ids)
            outputs.append(
                jnp.stack(
                    [
                        manual_read(model, context, query, pair)
                        for query, pair in zip(u, pairs, strict=True)
                    ]
                )
            )
        return jnp.stack(outputs)

    def result(forward):
        @eqx.filter_jit
        @eqx.filter_value_and_grad(has_aux=True)
        def loss(model, batch, target):
            output = forward(model, batch)
            return jnp.mean((output - target) ** 2), output

        return loss(model, batch, target)

    (loss, outputs), gradients = result(batched)
    (expected_loss, expected), expected_gradients = result(loop)
    close(loss, expected_loss)
    close(outputs, expected)
    for actual, reference in zip(
        jax.tree.leaves(gradients), jax.tree.leaves(expected_gradients), strict=True
    ):
        close(actual, reference)
    assert jnp.linalg.norm(gradients.pair_encoders[0].linear1.weight) > 0
    assert jnp.linalg.norm(gradients.context_blocks[0].attention.w_out) > 0
    assert jnp.linalg.norm(gradients.context_projection.weight) > 0
    assert jnp.linalg.norm(gradients.id_embedding.weight[episode[1]]) > 0
    np.testing.assert_array_equal(
        gradients.id_embedding.weight[jnp.array([0, 2, 5])], 0
    )
