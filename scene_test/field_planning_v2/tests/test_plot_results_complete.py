"""Regression: oracle Line2D must not displace the gray refusal bar legend."""
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

spec=importlib.util.spec_from_file_location("v2_plot_complete",Path(__file__).resolve().parents[1]/"plot_results_complete.py")
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)


def test_outcome_legend_includes_refusal_and_excludes_oracle_line_in_artist_order():
    fig,ax=plt.subplots()
    expected=("Returned / GT samples clear","Returned / GT sample invalid",
              "Returned / not scored","No path returned")
    bars=[ax.bar([0],[0],label=label) for label in expected]
    ax.axhline(18,label="Oracle passed GT samples: 18 / 20")
    # Matplotlib returns Line2D before BarContainer despite creation order.
    assert ax.get_legend_handles_labels()[1][0].startswith("Oracle")
    handles,labels=plot.outcome_legend(ax)
    assert labels==list(expected)
    assert handles==bars
    assert labels[-1]=="No path returned"
    plt.close(fig)


def test_outcome_legend_rejects_a_missing_category_instead_of_silently_omitting_it():
    fig,ax=plt.subplots()
    ax.bar([0],[1],label="Returned / GT samples clear")
    with pytest.raises(ValueError,match="exactly one outcome"):
        plot.outcome_legend(ax)
    plt.close(fig)
