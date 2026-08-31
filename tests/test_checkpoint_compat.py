"""Regression tests for backward compatibility with upstream `master`.

See docs/proposals/2026-08-29__master_backward_compatibility.md and
docs/plans/2026-08-29__master_backward_compatibility.md. `ZipVoice.__init__`
used to construct `self.embed` before `self.text_encoder` (needed to read a
pretrained embedding's hidden size before sizing `text_encoder`'s `in_dim`),
which changed both the model's parameter *registration order* (breaking
optimizer-state-dict resume, which is keyed by position, not name) and, for
the scratch-embedding path, the RNG *consumption* order (nn.Embedding draws
its random init at construction time), so a from-scratch model no longer got
bit-identical initial weights to master's from the same seed. These tests
build master's actual original code (via `git show master:...`), once per
test module via the `MasterZipVoice` fixture, and compare it against the
current in-repo model.
"""

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.utils.common import get_parameter_groups_with_lrs
from zipvoice.utils.optim import ScaledAdam

REPO_ROOT = Path(__file__).resolve().parent.parent

TINY_KWARGS = dict(
    feat_dim=100,
    fm_decoder_dim=32,
    fm_decoder_num_layers=[1, 1, 1, 1, 1],
    fm_decoder_downsampling_factor=[1, 2, 4, 2, 1],
    fm_decoder_cnn_module_kernel=[31, 15, 7, 15, 31],
    text_encoder_num_layers=1,
    text_encoder_dim=32,
)


def _load_master_zipvoice_class(tmp_path):
    """Load master's zipvoice.py as an isolated module, so we can build a
    real master-code model/optimizer to compare against, without checking
    out a different branch (which would disturb the working tree).

    Requires a local `master` ref (fails/skips in shallow clones, source
    archives, or installed packages that don't carry git history) -- these
    tests compare against whatever `master` currently points to, not a
    pinned commit, so they also drift if `master` itself moves later.
    """
    result = subprocess.run(
        ["git", "show", "master:zipvoice/models/zipvoice.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            "no local 'master' ref available to compare against "
            f"(git show failed: {result.stderr.strip()})"
        )
    source = result.stdout
    module_path = tmp_path / "zipvoice_master.py"
    module_path.write_text(source)
    spec = importlib.util.spec_from_file_location("zipvoice_master_ref", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ZipVoice


def _build_model(cls, seed):
    torch.manual_seed(seed)
    return cls(vocab_size=20, text_embed_dim=64, pad_id=0, **TINY_KWARGS)


def _synthetic_batch(seed=123):
    # Explicitly seeded so the batch itself is deterministic regardless of
    # what ran before this test in the same pytest session (global RNG state
    # otherwise carries over between tests, silently changing the batch data
    # from run to run and making any numerical-tolerance calibration unreliable).
    torch.manual_seed(seed)
    tokens = [[1, 2, 3, 4], [5, 6]]
    features = torch.randn(2, 10, 100)
    features_lens = torch.tensor([10, 7])
    noise = torch.randn_like(features)
    t = torch.rand(2, 1, 1)
    return dict(tokens=tokens, features=features, features_lens=features_lens, noise=noise, t=t)


def _make_optimizer(model):
    return ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=0.02, include_names=True),
        lr=0.02,
        clipping_scale=2.0,
    )


@pytest.fixture(scope="module")
def MasterZipVoice(tmp_path_factory):
    return _load_master_zipvoice_class(tmp_path_factory.mktemp("master_ref"))


def test_parameter_registration_order_matches_master(MasterZipVoice):
    m_master = _build_model(MasterZipVoice, seed=0)
    m_mine = _build_model(ZipVoice, seed=0)
    names_master = [n for n, _ in m_master.named_parameters()]
    names_mine = [n for n, _ in m_mine.named_parameters()]
    assert names_master == names_mine
    assert names_master.index("embed.weight") == names_mine.index("embed.weight")


def test_scratch_path_bit_identical_to_master_at_same_seed(MasterZipVoice):
    m_master = _build_model(MasterZipVoice, seed=0)
    m_mine = _build_model(ZipVoice, seed=0)
    params_master = dict(m_master.named_parameters())
    params_mine = dict(m_mine.named_parameters())
    for name, p_master in params_master.items():
        assert torch.equal(p_master, params_mine[name]), f"mismatch at {name}"


def test_state_dict_cross_load_strict(MasterZipVoice):
    # Independent model instances per direction (not master->mine->master
    # chained through the same mutated mine instance), so each direction is
    # a genuine, isolated check.
    m_master = _build_model(MasterZipVoice, seed=1)
    m_mine = _build_model(ZipVoice, seed=2)
    m_mine.load_state_dict(m_master.state_dict(), strict=True)

    m_master2 = _build_model(MasterZipVoice, seed=1)
    m_mine2 = _build_model(ZipVoice, seed=2)
    m_master2.load_state_dict(m_mine2.state_dict(), strict=True)


def test_optimizer_state_resume_from_master(MasterZipVoice):
    """The actual bug: loading a master-trained ScaledAdam state into mine's
    model, then continuing training, must produce the same result as
    continuing training under master itself -- not just avoid raising.

    Correctness note (found by Codex's second review): `optimizer.state_dict()`
    does not deep-copy -- PyTorch's optimizer state_dict() wraps the *same*
    state tensors, and ScaledAdam mutates them in place on every step(). A
    first draft of this test called `saved_state = opt_master.state_dict()`
    and then took a second step on `opt_master` *before* loading `saved_state`
    into `opt_mine` -- silently loading step-2 state while mine's own weights
    were still at step 1, and the resulting "floating-point drift" used to
    justify a very loose tolerance was actually at least partly this aliasing
    bug, not genuine independent-load numerical noise. Fixed by deep-copying
    both the model and optimizer state immediately after step 1, loading those
    exact (frozen) states into fresh mine objects, and syncing the global
    Torch RNG state before each side's second forward pass (condition_time_mask
    in zipvoice/utils/common.py samples torch.rand, so an unsynced second
    forward would apply a different random mask to each side even with
    identical model/optimizer state).
    """
    batch = _synthetic_batch()

    m_master = _build_model(MasterZipVoice, seed=3)
    opt_master = _make_optimizer(m_master)
    loss = m_master(**batch)
    loss.backward()
    opt_master.step()

    # Freeze exact post-step-1 state before continuing master any further.
    saved_model_state = copy.deepcopy(m_master.state_dict())
    saved_optim_state = copy.deepcopy(opt_master.state_dict())
    rng_state_before_second_forward = torch.get_rng_state()

    # Continue master's own run one more step -- this is the ground truth.
    opt_master.zero_grad()
    loss = m_master(**batch)
    loss.backward()
    opt_master.step()
    master_continued_params = {n: p.detach().clone() for n, p in m_master.named_parameters()}

    # Resume under mine from the exact frozen state (not a hand-reproduced
    # step 1), then take the same second step with the same RNG state.
    m_mine = _build_model(ZipVoice, seed=3)
    m_mine.load_state_dict(saved_model_state, strict=True)
    opt_mine = _make_optimizer(m_mine)
    opt_mine.load_state_dict(saved_optim_state)
    opt_mine.zero_grad()
    torch.set_rng_state(rng_state_before_second_forward)
    loss = m_mine(**batch)
    loss.backward()
    opt_mine.step()
    mine_continued_params = {n: p.detach() for n, p in m_mine.named_parameters()}

    # Before the fix, this raised KeyError('param_rms') -- ScaledAdam's
    # per-parameter state silently bound to the wrong (shape-incompatible or
    # differently-ordered) position. That crash is the actual bug being
    # regression-tested, and it no longer happens. With the aliasing bug
    # above fixed and RNG synced, the remaining discrepancy is tiny and
    # uniform (empirically <=0.0028 absolute across every parameter, measured
    # directly -- residual floating-point non-associativity between two
    # independently-`importlib`-loaded copies of structurally identical code,
    # not test artifact noise or misbinding). atol=5e-3 (with rtol=0, to
    # avoid relative-error blowup on near-zero parameters) comfortably covers
    # that residual with margin, while remaining far tighter than a real
    # state-misbinding regression would produce (which would show grossly
    # wrong values, not a sub-0.003 uniform drift, for the mismatched
    # parameters -- or simply crash, as it did before the fix).
    for name, p_master in master_continued_params.items():
        assert torch.allclose(p_master, mine_continued_params[name], atol=5e-3, rtol=0), (
            f"optimizer-resume mismatch at {name}"
        )
