"""agent_SpeedUp_v2 threshold policy: the Threshold record, the constants register and the T0 lint scaffold."""

from pathlib import Path

import pytest

from gpt_field import agent_speedup_v2_thresholds as th

LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
PACKAGE = Path(th.__file__).resolve().parent
# AC-1(c): the lint vocabulary, the allowlist and the part nouns are pinned; a change is a deliberate, reviewed
# version bump. D2 (2026-10-18): + PART_NOUNS, + allowlisted box / bin / flat / layer.
PINNED_VOCAB_SHA = "2542b2d9cd83b34b76547712695f3219c1cb2ea70e80374188007d8e29b747b7"
PINNED_ALLOWLIST_SHA = "fe1c5da7b24dcd75c1a04fec1fbcf94f2340b5a748cd0483b7e6d0c548316f92"
PINNED_PART_NOUNS_SHA = "4172f284104e7fefb03cf350e89b893a3aed1b60ea9a7ebd64e35a965fc1837d"


@pytest.fixture
def libero_checkout():
    """The lint tests never skip: a missing checkout would leave AC-1(c) unguarded."""
    if not Path(LIBERO, "libero/libero/bddl_files").is_dir():
        pytest.fail("LIBERO checkout required for AC-1(c) (the T0 lint vocabulary)")
    return LIBERO


needs_bddl = pytest.mark.usefixtures("libero_checkout")


def test_threshold_record():
    t = th.Threshold("G5_object_present", 0.5, "DD", "max(0.5 mm, 4 sigma_close)")
    assert t.to_dict() == {"name": "G5_object_present", "value": 0.5, "label": "DD",
                           "derivation": "max(0.5 mm, 4 sigma_close)"}
    for bad in (dict(label="TUNED"), dict(derivation=" "), dict(value=float("nan")), dict(name="")):
        kwargs = {"name": "x", "value": 1.0, "label": "ROBOT", "derivation": "d", **bad}
        with pytest.raises(ValueError):
            th.Threshold(**kwargs)


def test_constants_register():
    register = th.constants_register()
    assert register and all(entry["label"] in th.LABEL_CLASSES and entry["derivation"]
                            for entry in register.values())
    assert th.constants_sha256() == th.constants_sha256() and len(th.constants_sha256()) == 64
    assert not any("TUNED" in entry["label"] for entry in register.values())


@needs_bddl
def test_vocabulary():
    vocab = th.lint_vocabulary(LIBERO)
    words = set(vocab["vocabulary"])
    assert {"stove", "moka", "pot", "bowl", "drawer", "plate", "cabinet", "basket", "microwave"} <= words
    assert not words & set(th.LINT_ALLOWLIST)
    assert vocab["bddl_files"] > 100
    assert vocab["vocabulary_sha256"] == PINNED_VOCAB_SHA, "lint vocabulary changed: review and re-pin"
    assert vocab["allowlist_sha256"] == PINNED_ALLOWLIST_SHA, "lint allowlist changed: review and re-pin"
    assert vocab["part_nouns_sha256"] == PINNED_PART_NOUNS_SHA, "part nouns changed: review and re-pin"
    assert len(vocab["vocabulary"]) == 66
    # D2: part nouns are banned, generic programming words and the cavity 'rim' are allowed
    assert set(th.PART_NOUNS) <= words and vocab["part_nouns"] == sorted(th.PART_NOUNS)
    assert {"handle", "knob", "lid", "burner", "lever", "door", "button", "spout"} <= words
    assert not {"box", "bin", "flat", "layer", "rim"} & words
    assert not set(th.PART_NOUNS) & set(th.LINT_ALLOWLIST)
    assert all(len(word) >= th.MIN_TOKEN_CHARS for word in th.LINT_ALLOWLIST)     # no dead entries


@needs_bddl
def test_t0_lint_passes_on_v2_control_modules():
    report = th.lint_control_modules(LIBERO)
    assert report["passed"], report["modules"]
    assert set(report["modules"]) == set(th.CONTROL_MODULES)


@needs_bddl
def test_t0_lint_catches_violations(tmp_path):
    vocab = th.lint_vocabulary(LIBERO)["vocabulary"]
    planted = tmp_path / "planted.py"
    planted.write_text('import math\nRIM_OFFSET_MM = 3.0\nLABELLED = 1.0\nANGLE = math.radians(5.0)\n'
                       'def approach_bowl(task_prompt):\n    return "the moka handle"\n')
    labels = {"planted.LABELLED": {"label": "DD", "derivation": "x"}}
    found = th.lint_module(planted, vocab, labels=labels)
    rules = {(v["rule"], v.get("name") or v.get("text")) for v in found}
    assert ("unlabelled_constant", "planted.RIM_OFFSET_MM") in rules
    assert ("unlabelled_constant", "planted.ANGLE") in rules
    assert ("unlabelled_constant", "planted.LABELLED") not in rules
    assert any(v["rule"] == "lint_vocabulary" and "bowl" in v["tokens"] for v in found)
    assert any(v["rule"] == "lint_vocabulary" and "moka" in v["tokens"] for v in found)
    assert any(v["rule"] == "task_text_identifier" for v in found)
    assert not any(v["rule"] == "lint_vocabulary" and "rim" in v["tokens"] for v in found)   # allowlisted


PLANTED = {
    # TLR-1: numeric constants in any expression form, class attributes and function defaults
    "E_dict": ("LIMITS = {'gap_rise_mm': 0.37, 'stall': 0.4}\n", ("unlabelled_constant", "planted.LIMITS")),
    "F_default": ("def guard(x, limit=0.123):\n    return x > limit\n", ("numeric_default", "planted.guard.limit")),
    "F_kwonly": ("def guard(x, *, limit=-2):\n    return x\n", ("numeric_default", "planted.guard.limit")),
    "H_class": ("class Cfg:\n    MARGIN = 4.2\n", ("unlabelled_constant", "planted.Cfg.MARGIN")),
    "I_kwcall": ("THR = dict(g=0.5)\n", ("unlabelled_constant", "planted.THR")),
    "J_ifexp": ("SWITCH = 3.0 if True else 4.0\n", ("unlabelled_constant", "planted.SWITCH")),
    # TLR-2: imports
    "M_alias_attr": ("import gpt_field.agent_speedup_v1 as v1\nZ = v1.DRAWER_PULL_M\n", ("lint_vocabulary", "drawer")),
    "N_from_asname": ("from gpt_field.agent_speedup_v1 import BOWL_RIM_MM as m\n", ("lint_vocabulary", "bowl")),
    "O_from_module": ("from gpt_field.bowl_helpers import lift as lf\n", ("lint_vocabulary", "bowl")),
    "W_evaluator": ("from .agent_speedup_v2_truth import evaluate_goal as et\n", ("forbidden_import", None)),
    "W_libero": ("from gpt_field import libero\n", ("forbidden_import", None)),
    # TLR-3: plurals, compounds and all-caps runs
    "K_plural": ("def stack_bowls(plates, drawers, mugs):\n    return plates\n", ("lint_vocabulary", "plate")),
    "L_compound": ("def mokapot_lift(stovetop):\n    return stovetop\n", ("lint_vocabulary", "stove")),
    "S_allcaps": ("BOWLRIM = 'x'\n", ("lint_vocabulary", "bowl")),
    # TLR-5 / D2: part nouns are vocabulary; the vocabulary definitions are exempt only in the lint's own module
    "Q_part_nouns": ("def twist_knob(handle, lid, burner, lever, door):\n    return 'spout button'\n",
                     ("lint_vocabulary", "knob")),
    "Q_part_plural": ("def pull_handles(levers):\n    return levers\n", ("lint_vocabulary", "lever")),
    "Q_definition_elsewhere": ("PART_NOUNS = ('handle',)\nLINT_ALLOWLIST = ('bowl',)\n",
                               ("lint_vocabulary", "handle")),
}


@needs_bddl
@pytest.mark.parametrize("name", sorted(PLANTED))
def test_t0_lint_catches_planted_cases(name, tmp_path):
    vocab = th.lint_vocabulary(LIBERO)["vocabulary"]
    source, (rule, needle) = PLANTED[name]
    planted = tmp_path / "planted.py"
    planted.write_text(source)
    found = th.lint_module(planted, vocab, labels={})
    assert any(v["rule"] == rule and (needle is None or needle == v.get("name") or needle in v.get("tokens", ()))
               for v in found), found
    if name.startswith(("M_", "N_")):
        assert any(v["rule"] == "forbidden_import" for v in found)     # v1 modules are never imported


@needs_bddl
def test_t0_lint_generic_code_stays_clean(tmp_path):
    """No false positives on generic geometry code (allowlisted words, numpy attributes, labelled constants)."""
    vocab = th.lint_vocabulary(LIBERO)["vocabulary"]
    planted = tmp_path / "planted.py"
    planted.write_text("import numpy as np\nfrom dataclasses import dataclass\nGAP_MM = 1.0\n"
                       "def rim_floor_target_object(rim_z, floor_z, regions, objects):\n"
                       "    return np.linalg.norm(rim_z - floor_z)\n")
    assert th.lint_module(planted, vocab, labels={"planted.GAP_MM": {"label": "DD", "derivation": "x"}}) == []


@needs_bddl
def test_t0_lint_generic_programming_words_allowed(tmp_path):
    """D2: bounding box, histogram bin, flat plane fill, HeightField layer are generic; 'table' stays banned."""
    vocab = th.lint_vocabulary(LIBERO)["vocabulary"]
    planted = tmp_path / "planted.py"
    planted.write_text("def bounding_box(hist_bin, flat_patch, layer, boxes, bins, layers):\n"
                       "    return 'bounding box of the flat layer'\n")
    assert th.lint_module(planted, vocab, labels={}) == []
    planted.write_text("def bounding_box(lookup_table):\n    return lookup_table\n")
    found = th.lint_module(planted, vocab, labels={})
    assert found and all(v["tokens"] == ["table"] for v in found)


def test_token_hits():
    vocab = {"bowl", "plate", "moka", "pot", "stove", "box"}
    known = {"rim", "top", "in"}
    assert th.token_hits("bowls", vocab) == {"bowl"} and th.token_hits("boxes", vocab) == {"box"}
    assert th.token_hits("mokapot", vocab) == {"moka", "pot"} and th.token_hits("bowlrim", vocab, known) == {"bowl"}
    assert th.token_hits("stovetop", vocab, known) == {"stove"}
    assert th.token_hits("template", vocab, known) == set() and th.token_hits("into", vocab, known) == set()


def test_every_control_module_constant_is_labelled():
    import ast
    for name in th.CONTROL_MODULES:
        tree = ast.parse((PACKAGE / name).read_text())
        for constant in th.module_numeric_constants(tree):
            assert f"{Path(name).stem}.{constant}" in th.CONSTANT_LABELS, (name, constant)
        for default in th.numeric_defaults(tree):
            assert f"{Path(name).stem}.{default}" in th.CONSTANT_LABELS, (name, default)
        assert th.forbidden_imports(tree) == [], name


def test_tokens():
    assert th.tokens("flat_stove_1_cook_region") == ["flat", "stove", "cook", "region"]
    assert th.tokens("MokaPot handle") == ["moka", "pot", "handle"]
