import math

import torch

from demo import tinygpt


def _data():
    return tinygpt.encode("To be, or not to be, that is the question. " * 200)


def test_vocab_is_fixed_and_unknown_maps_to_question_mark():
    ids = tinygpt.encode("a\né")
    assert ids.dtype == torch.long
    assert int(ids.max()) < tinygpt.VOCAB_SIZE
    assert tinygpt.decode(ids) == "a\n?"


def test_batch_for_step_is_deterministic_and_varies_by_step():
    d = _data()
    x1, y1 = tinygpt.batch_for_step(d, step=5, B=8, ctx=32)
    x2, y2 = tinygpt.batch_for_step(d, step=5, B=8, ctx=32)
    x3, _ = tinygpt.batch_for_step(d, step=6, B=8, ctx=32)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert not torch.equal(x1, x3)
    assert x1.shape == (8, 32) and torch.equal(x1[:, 1:], y1[:, :-1])


def test_fresh_model_loss_is_about_log_vocab():
    torch.manual_seed(0)
    m = tinygpt.TinyGPT(ctx=32)
    x, y = tinygpt.batch_for_step(_data(), 0, 8, 32)
    loss = tinygpt.loss_fn(m, x, y)
    assert abs(loss.item() - math.log(tinygpt.VOCAB_SIZE)) < 0.5


def test_accumulation_equals_full_batch_gradient():
    torch.manual_seed(0)
    m = tinygpt.TinyGPT(ctx=32)
    x, y = tinygpt.batch_for_step(_data(), 3, 12, 32)
    m.zero_grad()
    tinygpt.loss_fn(m, x, y).backward()
    full = tinygpt.flat_grads(m).clone()
    m.zero_grad()
    loss = tinygpt.accumulate_grads(m, x, y, cap=4)
    acc = tinygpt.flat_grads(m)
    assert torch.allclose(full, acc, atol=1e-6, rtol=1e-4)
    assert abs(loss - float(tinygpt.loss_fn(m, x, y))) < 1e-5


def test_flat_grads_roundtrip_and_checksum():
    torch.manual_seed(0)
    a, b = tinygpt.TinyGPT(ctx=32), tinygpt.TinyGPT(ctx=32)
    assert tinygpt.checksum(a) != tinygpt.checksum(b)
    b.load_state_dict(a.state_dict())
    assert tinygpt.checksum(a) == tinygpt.checksum(b)
    x, y = tinygpt.batch_for_step(_data(), 0, 4, 32)
    tinygpt.loss_fn(a, x, y).backward()
    g = tinygpt.flat_grads(a)
    tinygpt.set_flat_grads(b, g * 2)
    assert torch.allclose(tinygpt.flat_grads(b), g * 2)


def test_empty_slice_gives_zero_grads():
    m = tinygpt.TinyGPT(ctx=32)
    x, y = tinygpt.batch_for_step(_data(), 0, 4, 32)
    loss = tinygpt.accumulate_grads(m, x[:0], y[:0], cap=4)
    assert loss is None
    assert float(tinygpt.flat_grads(m).abs().sum()) == 0.0
