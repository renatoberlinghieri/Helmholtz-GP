import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import numpy as np
from typing import Dict, List
import json_tricks
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams["figure.figsize"] = [4, 2]
mpl.rcParams["figure.dpi"] = 300
np.set_printoptions(precision=5)
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 12,
    }
)


def load_data(directory: Path) -> Dict:
    """
    Load the dataset specified by dataset_name
    """
    load_path = str(Path(directory, f"lasersparse.json"))
    return json_tricks.load(load_path)


def load_results(results_dir: Path) -> Dict:
    methods = ["standard", "helmholtz"]
    load_paths = [
        str(Path(results_dir, "lasersparse", f"{method}.json")) for method in methods
    ]
    return [json_tricks.load(load_path) for load_path in load_paths]


def plot_data_and_splits(dataset: Dict, all_results: List[Dict]):
    XY_train = dataset["XY"]
    UV_train = dataset["UV"]
    cmap = cm.get_cmap("viridis")
    all_rmses = [
        v["metrics"]["rmse"] if k.startswith("split_") else 0
        for results in all_results for k, v in results.items() 
    ]
    import pdb; pdb.set_trace()
    max_rmse = np.max(all_rmses)

    min_long = np.min(XY_train[:, 0])
    max_long = np.max(XY_train[:, 0])
    min_lat = np.min(XY_train[:, 1])
    max_lat = np.max(XY_train[:, 1])

    split_long = 3
    split_lat = 3
    offset_long = (abs(max_long - min_long)) / split_long
    offset_lat = (abs(max_lat - min_lat)) / split_lat
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    methods = ["Standard", "Helmholtz"]
    for results, ax, method in zip(all_results, axes, methods):
        ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:, 0], UV_train[:, 1])
        ax.vlines(min_long + offset_long, min_lat, max_lat, colors="k")
        ax.vlines(min_long + 2 * offset_long, min_lat, max_lat, colors="k")
        ax.hlines(min_lat + offset_lat, min_long, max_long, colors="k")
        ax.hlines(min_lat + 2 * offset_lat, min_long, max_long, colors="k")
        ax.set_xlim(min_long, max_long)
        ax.set_ylim(min_lat, max_lat)
        ax.set_title(method)


        for split_name, result in results.items():
            if not split_name.startswith("split"):
                continue
            split_number = int(split_name.split("_")[-1])
            rmse = result["metrics"]["rmse"]

            # Get bounds of box
            left_bound = min_long + (split_number % split_long) * offset_long
            right_bound = left_bound + offset_long
            upper_bound = max_lat - (split_number // split_lat) * offset_lat
            lower_bound = upper_bound - offset_lat

            yy = np.linspace(lower_bound, upper_bound, 100)

            ax.fill_betweenx(
                yy,
                left_bound,
                right_bound,
                color=cmap(rmse / max_rmse),
                alpha=0.5,
            )

    norm = mpl.colors.Normalize(vmin=0, vmax=max_rmse)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb1 = mpl.colorbar.ColorbarBase(
        cax,
        cmap=cmap,
        norm=norm,
    )


if __name__ == "__main__":
    experiments_dir = Path(__file__).parent.parent
    data_dir = Path(experiments_dir.parent, "data")
    results_dir = Path(experiments_dir, "results")
    # Load data and results
    data = load_data(data_dir)
    all_results = load_results(results_dir)
    plot_data_and_splits(data, all_results)
    plt.tight_layout()
    plt.show()
