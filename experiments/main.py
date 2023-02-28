import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import argparse

from pathlib import Path
from typing import Dict, List, Iterable, NamedTuple, Callable
from numpy.typing import ArrayLike
import json_tricks
import helmholtz_gp.helmholtz_regression_pytorch as hrp
from helmholtz_gp.parameters import TwoKernelGPParams, initial_parameters
from helmholtz_gp.optimization_loop import basic_optimization_loop
from helmholtz_gp.utils import _to_numpy, _to_torch, _list_to_numpy
from collections import namedtuple

Dataset = namedtuple("Dataset", ["XY_train", "XY_test", "UV_train", "UV_test"])


def setup_args() -> argparse.Namespace:
    """
    Handle user arguments passed with argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Dataset name")
    parser.add_argument(
        "-m", "--method", help="Method name", choices=["standard", "helmholtz"]
    )
    parser.add_argument(
        "-n", "--num_iter", help="Number of iterations for Adam", default=1000
    )
    parser.add_argument("--cross_validation", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def load_data(dataset_name: str, cross_validation: bool = False) -> Dict:
    """
    Load the dataset specified by dataset_name
    """
    data_dir = Path(Path(__file__).parent.parent, "data")
    data = dict()  # init dictionary to return
    for arr in ["XY_train", "XY_test", "UV_train", "UV_test"]:
        load_path = Path(data_dir, f"{arr}_{args.dataset}.csv")
        data[arr] = np.loadtxt(str(load_path), delimiter=",")
    return _to_torch(data)


def build_loss(method: str, data: Dataset) -> Callable:
    def _loss(params):
        return -hrp.lml(data.XY_train, data.UV_train, kind=method, params=params)

    return _loss


def build_data_iterator(data_dict: Dict) -> Iterable[Dataset]:
    train_splits = data_dict["train_splits"]  # Get array of splits representing train
    test_splits = data_dict["test_splits"]  # Get array of test splits
    num_splits = len(train_splits)
    assert (
        len(test_splits) == num_splits
    )  # Should have same number of test/train splits
    for i in range(num_splits):
        train_inds = train_splits[i]
        test_inds = test_splits[i]
        XY_train = data_dict["XY"][train_inds]
        XY_test = data_dict["XY"][test_inds]
        UV_train = data_dict["UV"][train_inds]
        UV_test = data_dict["UV"][test_inds]
        split = Dataset(XY_train, XY_test, UV_train, UV_test)
        yield split  # Iterable over splits


def compute_metrics(
    post_mean: torch.Tensor, post_cov: torch.Tensor, UV_test: torch.Tensor
) -> Dict:
    post_mean_reshaped = torch.concat(
        [post_mean[: len(post_mean) // 2], post_mean[len(post_mean) // 2 :]], axis=-1
    )
    sq_diffs = torch.square(post_mean_reshaped - UV_test)
    velocity_sq_diffs = sq_diffs.sum(-1)
    mean_sq_diffs = velocity_sq_diffs.mean()
    rmse = torch.sqrt(mean_sq_diffs)
    metrics_dict = dict(rmse=rmse.detach().numpy())
    return metrics_dict


def add_total_metric_info(results: Dict, metric_names: List = ["rmse"]) -> Dict:
    results["summary_metrics"] = dict()
    for metric in metric_names:
        all_vals = list()
        for k, v in results.items():
            if k.startswith("split_"):
                all_vals.append(v["metrics"][metric])
                metric_mean = np.mean(all_vals)
                metric_std = np.std(all_vals)
                results["summary_metrics"][metric]["mean"] = metric_mean
                results["summary_metrics"][metric]["std"] = metric_std

    return results


def train_and_predict(args: argparse.Namespace, dataset: Dataset) -> Dict:
    params = initial_parameters()[args.method][args.dataset]  # (re)set parameters
    loss = build_loss(args.method, dataset)  # Build the loss for the current dataset
    basic_optimization_loop(
        loss, params, num_iter=int(args.num_iter)
    )  # approximate MLL
    post_mean, post_cov, _ = hrp.posterior_kernel_twodata(  # Compute predictive
        dataset.XY_test,
        dataset.XY_test,
        dataset.UV_train,
        dataset.XY_train,
        args.method,
        params,
    )

    metrics = compute_metrics(post_mean, post_cov, dataset.UV_test)  # Compute metrics
    return dict(
        best_params=_list_to_numpy(params.get_params()),
        post_mean=post_mean,
        post_cov=post_cov,
        metrics=metrics,
    )


if __name__ == "__main__":
    args = setup_args()  # Setup arguments
    data = load_data(args.dataset)  # Load data, this is a dictionary
    results = dict()  # initialize a dictionary, we will add to it later

    if args.cross_validation:
        splits = build_data_iterator(data)  # Make an iterable to loop over
        for i, dataset in enumerate(splits):  # Loop over test train splits
            new_dict_entry = train_and_predict(args, dataset)
            results[f"split_{i}"] = new_dict_entry
        add_total_metric_info(results)  # Add summary statistics

    else:
        dataset = Dataset(
            data["XY_train"], data["XY_test"], data["UV_train"], data["UV_test"]
        )
        results = train_and_predict(args, dataset)

    # Save results
    results_dir = Path(Path(__file__).parent, "results", args.dataset)
    os.makedirs(results_dir, exist_ok=True)  # Make directory if doesn't exist
    results_path = Path(results_dir, f"{args.method}.json")  # Build full path to save
    json_tricks.dump(
        _to_numpy(results), str(results_path)
    )  # dump dictionary, make sure no tensors
