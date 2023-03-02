import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import json_tricks
import numpy as np


# set plotting params for the notebook
def set_plotting_params():
    mpl.rcParams["figure.figsize"] = [20, 10]
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


if __name__ == "__main__":
    set_plotting_params()
    data_path = str(Path(__file__).parent)
    X_grid = np.genfromtxt(f"{data_path}/Xgrid_lasersparse.csv", delimiter=",")
    Y_grid = np.genfromtxt(f"{data_path}/Ygrid_lasersparse.csv", delimiter=",")
    XY_train = np.genfromtxt(f"{data_path}/XY_train_lasersparse.csv", delimiter=",")

    XY_test = np.genfromtxt(f"{data_path}/XY_test_lasersparse.csv", delimiter=",")

    UV_train = np.genfromtxt(f"{data_path}/UV_train_lasersparse.csv", delimiter=",")
    min_long = np.min(XY_train[:, 0])
    max_long = np.max(XY_train[:, 0])
    min_lat = np.min(XY_train[:, 1])
    max_lat = np.max(XY_train[:, 1])
    split_long = 3
    split_lat = 3
    offset_long = (abs(max_long - min_long)) / split_long
    offset_lat = (abs(max_lat - min_lat)) / split_lat

    plt.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:, 0], UV_train[:, 1])
    p1, p2 = [min_long, max_long], [max_lat - offset_lat, max_lat - offset_lat]
    p3, p4 = [min_long, max_long], [max_lat - 2 * offset_lat, max_lat - 2 * offset_lat]
    p5, p6 = [min_long + offset_long, min_long + offset_long], [max_lat, min_lat]
    p7, p8 = [min_long + 2 * offset_long, min_long + 2 * offset_long], [
        max_lat,
        min_lat,
    ]
    plt.plot(p1, p2, p3, p4, marker="o", c="red")
    plt.plot(p5, p6, p7, p8, marker="o", c="green")

    lat_splits = [max_lat - i * offset_lat for i in range(split_lat + 1)]
    long_splits = [min_long + i * offset_long for i in range(split_long + 1)]

    train_splits = list()
    test_splits = list()
    for i in range(len(lat_splits) - 1):
        for j in range(len(long_splits) - 1):
            mask_long = (XY_train[:, 0] >= long_splits[j]) & (
                XY_train[:, 0] < long_splits[j + 1]
            )
            mask_lat = (XY_train[:, 1] <= lat_splits[i]) & (
                XY_train[:, 1] > lat_splits[i + 1]
            )
            mask = mask_long & mask_lat
            test_splits.append(mask)
            train_splits.append(~mask)
    train_splits = [np.where(split)[0] for split in train_splits]
    test_splits = [np.where(split)[0] for split in test_splits]

    data_dict = {
        "train_splits": train_splits,
        "test_splits": test_splits,
        "XY": XY_train,
        "UV": UV_train,
    }
    save_path = str(Path(data_path, "lasersparse.json"))
    json_tricks.dump(data_dict, save_path)
