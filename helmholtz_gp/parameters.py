import torch
from dataclasses import dataclass
import numpy as np

data_type = torch.float32


@dataclass
class TwoKernelGPParams:
    log_ls_1: torch.Tensor = torch.tensor(0.1, dtype=data_type)
    log_sigma_1: torch.Tensor = torch.tensor(1.0, dtype=data_type)
    log_ls_2: torch.Tensor = torch.tensor(0.1, dtype=data_type)
    log_sigma_2: torch.Tensor = torch.tensor(1.0, dtype=data_type)
    log_obs_noise: torch.Tensor = torch.tensor(0.02, dtype=data_type)

    @property
    def ls_1(self):
        return torch.exp(self.log_ls_1)

    @property
    def sigma_1(self):
        return torch.exp(self.log_sigma_1)

    @property
    def ls_2(self):
        return torch.exp(self.log_ls_2)

    @property
    def sigma_2(self):
        return torch.exp(self.log_sigma_2)

    @property
    def obs_noise(self):
        return torch.exp(self.log_obs_noise)

    def get_unconstrained_params(self):
        return (
            self.log_ls_1,
            self.log_sigma_1,
            self.log_ls_2,
            self.log_sigma_2,
            self.log_obs_noise,
        )

    def get_params(self):
        return [torch.exp(p) for p in self.get_unconstrained_params()]


synthetic_datasets = [
    "point_divergence",
    "vortex",
    "continuity_vortex",
    "continuous_current_mild",
    "continuous_current_weak",
    "continuous_current_strong",
    "duffing_divergence",
]


def synthetic_default_parameters():
    log_ls_1 = torch.tensor(0.0, requires_grad=True)
    log_sigma_1 = torch.tensor(0.0, requires_grad=True)
    log_ls_2 = torch.tensor(1.0, requires_grad=True)
    log_sigma_2 = torch.tensor(-1.0, requires_grad=True)
    log_obs_noise = torch.tensor(-2.0, requires_grad=True)
    return TwoKernelGPParams(
        log_ls_1, log_sigma_1, log_ls_2, log_sigma_2, log_obs_noise
    )


def GLAD_helmholtz_default_parameters():
    log_ls_Phi = torch.tensor(2.5, requires_grad=True)
    log_sigma_Phi = torch.tensor(-2.0, requires_grad=True)
    log_ls_A = torch.tensor(2.0, requires_grad=True)
    log_sigma_A = torch.tensor(1.1, requires_grad=True)
    log_obs_noise = torch.tensor(-2.0, requires_grad=True)
    return TwoKernelGPParams(
        log_ls_Phi, log_sigma_Phi, log_ls_A, log_sigma_A, log_obs_noise
    )


def GLAD_standard_default_parameters():
    log_ls_u = torch.tensor(1.0, requires_grad=True)
    log_sigma_u = torch.tensor(0.0, requires_grad=True)
    log_ls_v = torch.tensor(1.0, requires_grad=True)
    log_sigma_v = torch.tensor(0.0, requires_grad=True)
    log_obs_noise = torch.tensor(-1.0, requires_grad=True)
    return TwoKernelGPParams(
        log_ls_u, log_sigma_u, log_ls_v, log_sigma_v, log_obs_noise
    )


initial_parameters = dict(helmholtz=dict(), standard=dict())

for dataset_name in synthetic_datasets:
    initial_parameters["helmholtz"][dataset_name] = synthetic_default_parameters()
    initial_parameters["standard"][dataset_name] = synthetic_default_parameters()
initial_parameters["helmholtz"]["GLAD_full"] = GLAD_helmholtz_default_parameters()
initial_parameters["standard"]["GLAD_full"] = GLAD_standard_default_parameters()
