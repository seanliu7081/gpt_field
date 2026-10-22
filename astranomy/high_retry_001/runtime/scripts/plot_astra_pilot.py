"""Render returned field slices and source cameras; no API calls or simulation."""

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    args = parser.parse_args()
    root = Path(args.run)
    observation = json.loads((root / "observation.json").read_text())
    samples = [json.loads(p.read_text()) for p in sorted(root.glob("sample_*.field.json"))]
    if not samples:
        raise ValueError("No returned fields to plot")
    fig, axes = plt.subplots(len(samples)+1, 3, figsize=(12, 3.3*(len(samples)+1)), squeeze=False,
                             layout="constrained")
    for column, key in enumerate(("observation/image", "observation/wrist_image")):
        raw = base64.b64decode(observation[key].split(",",1)[1])
        axes[0,column].imshow(Image.open(io.BytesIO(raw)))
        axes[0,column].set_title("External camera" if column==0 else "Wrist camera")
        axes[0,column].axis("off")
    axes[0,2].axis("off")
    axes[0,2].text(0,1,
        "Astra static-observation pilot\n\n" + observation["prompt"] +
        "\n\nOne unchanged observation; repeated requests.\nLower potential is preferred.\n"
        "Color scales are per sample.\nValues are unitless, not meters.\nNo robot actions were executed.",
        va="top",wrap=True,fontsize=10)
    for sample_index, sample in enumerate(samples):
        size = sample["grid"]["size"]
        spacing_cm = sample["grid"]["spacing_m"]*100
        pred = sample["prediction"]
        values = np.array(pred["values"],dtype=float).reshape((size,)*3)
        masked = values.copy().reshape(-1)
        masked[pred["blocked_indices"]+pred["unknown_indices"]] = np.nan
        masked = masked.reshape((size,)*3)
        center = size//2
        limit = spacing_cm*(size-1)/2
        extent = [-limit-spacing_cm/2,limit+spacing_cm/2]*2
        limits = (float(np.nanmin(masked)),float(np.nanmax(masked)))
        planes = [(masked[:,:,center].T,"x","y","z"),
                  (masked[:,center,:].T,"x","z","y"),
                  (masked[center,:,:].T,"y","z","x")]
        for column,(plane,horizontal,vertical,fixed) in enumerate(planes):
            axis = axes[sample_index+1,column]
            artist = axis.imshow(plane,origin="lower",extent=extent,cmap="viridis_r",
                                 interpolation="nearest",vmin=limits[0],vmax=limits[1])
            axis.plot(0,0,"r+",markersize=12,markeredgewidth=2)
            axis.set_title(f"Sample {sample_index+1}: {horizontal}{vertical} slice, {fixed} at center")
            axis.set_xlabel(f"World {horizontal} offset (cm)")
            axis.set_ylabel(f"World {vertical} offset (cm)")
            axis.set_xticks([-8,-4,0,4,8])
            axis.set_yticks([-8,-4,0,4,8])
            fig.colorbar(artist,ax=axis,label="Unitless potential",shrink=0.8)
    destination = root/"field_slices.png"
    fig.savefig(destination,dpi=150)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()

