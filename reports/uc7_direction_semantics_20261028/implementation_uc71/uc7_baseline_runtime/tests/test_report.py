import pytest

from gpt_field.report import distance_fidelity, summarize


def row(method, task, index, status, protocol="same"):
    return dict(method=method,task_id=task,episode_index=index,status=status,
                suite="libero_10",seed=7,protocol_hash=protocol)


def test_task_balance_and_pairing():
    rows = [row("a",0,0,"success"),row("a",0,1,"timeout"),row("a",1,0,"success"),
            row("b",0,0,"timeout"),row("b",0,1,"timeout"),row("b",1,0,"success")]
    result = summarize(rows)
    assert result["methods"]["a"]["task_mean_success_rate"] == 0.75
    assert result["paired"][0]["pairs"] == 3
    assert result["paired"][0]["task_mean_difference_a_minus_b"] == 0.25
    assert sum(result["methods"]["a"]["statuses"].values()) == 3


def test_mismatched_protocol_not_paired_and_duplicates_rejected():
    assert not summarize([row("a",0,0,"success","one"),row("b",0,0,"success","two")])["paired"]
    with pytest.raises(ValueError,match="duplicate"):
        summarize([row("a",0,0,"success")]*2)
    mixed = [row("a",0,0,"success"),row("b",0,0,"success")]
    mixed[1]["suite"] = "libero_goal"
    with pytest.raises(ValueError,match="one benchmark suite"):
        summarize(mixed)


def test_fidelity_cannot_claim_potential_is_metric():
    with pytest.raises(ValueError,match="semantics"):
        distance_fidelity({"predicted":[1],"reference":[2]})
    result = distance_fidelity({"semantics":"distance_m","predicted":[0,0.02],"reference":[0,0.01]})
    assert result["mae_m"] == pytest.approx(0.005)
