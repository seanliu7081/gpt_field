"""Display true and reconstructed thin-wall cross sections from a saved run."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sdf_grid import SDFGrid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "metrics.json").read_text())
    if report["status"] not in ("complete", "completed_with_failures"):
        raise ValueError("Wait for the reconstruction run to finish")
    spec = importlib.util.spec_from_file_location("frozen_plot_geometry", run / "source" / "geometry.py")
    geometry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = geometry
    spec.loader.exec_module(geometry)
    scenes = [s for s in geometry.make_scenes() if s.category == "thin_walls_and_narrow_gaps"]
    rows = {(r["scene"], r["spacing_m"]): r for r in report["configurations"]}
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey=True, constrained_layout=True)
    for i, scene in enumerate(scenes):
        section = scene.cross_sections[0]
        parameter = np.linspace(-.008, .008, 4001)
        points = np.asarray(section["origin_m"]) + parameter[:, None]*np.asarray(section["direction"])
        truth = scene.sdf(points)
        expected = np.array(section["expected_zero_crossings_m"])*1000
        for j, h in enumerate((.004, .002, .001)):
            ax = axes[i, j]
            row = rows.get((scene.name, h))
            if not row or row["status"] == "failed":
                ax.text(.5, .5, "Failed / missing", transform=ax.transAxes, ha="center")
                continue
            grid = SDFGrid.load(run / row["artifacts"]["field"])
            prediction = grid.query(points)
            ax.axvspan(expected[0], expected[1], color="gray", alpha=.23)
            ax.axvspan(expected[2], expected[3], color="gray", alpha=.23)
            ax.axhline(0, color="gray", lw=.7)
            ax.plot(parameter*1000, truth*1000, color="black", label="True signed distance")
            ax.plot(parameter*1000, prediction*1000, color="#ca3535", label="Grid interpolation")
            wrong = (truth < 0) != (prediction < 0)
            ax.scatter(parameter[wrong][::12]*1000, np.full(wrong.sum(), -2.8)[::12],
                       color="#b90088", s=6, label="Wrong material/air sign")
            feature = row["features"]["cross_sections"][0]
            ax.set_title(f"Seed {scene.seed}; grid {h*1000:g} mm\n"
                         f"surface crossings: {len(feature['predicted_zero_crossings_m'])} / 4")
            ax.set(xlim=(-8, 8), ylim=(-3, 5), xlabel="Local cross-section position (mm)",
                   ylabel="Signed surface distance (mm)")
            ax.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("Thin walls and gaps: gray bands are true solid material\n"
                 "Negative distance = solid; positive = air. One fixed local cross section per scene.")
    out = run / "figures" / "thin_features.png"
    if out.exists():
        raise FileExistsError(out)
    fig.savefig(out, dpi=170)
    print(out)


if __name__ == "__main__":
    main()
