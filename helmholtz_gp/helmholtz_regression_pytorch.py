# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import torch

# Hide any GPUs to that PyTorch uses CPU (typically preferable due to memory constraints)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# First, define function for kernel
def sqr_exp_2D(X1, Y1, X2, Y2, ls, sigma):
    """Sqr_exp_2D computes the covariances matrix for (X1, Y1) with (X2, Y2) under a squared exponential 
    kernel with a single length scale.
    
    Args:
        X1, Y1, X2, Y2 : torch.Tensor of shape [N, M]
        ls: length scale
        sigma: signal variance
    """
    d_sqr = X1**2 + X2**2 - 2*X1*X2 + Y1**2+Y2**2-2*Y1*Y2
    K = sigma**2 * torch.exp(-(1./2.)*(d_sqr)/ls**2)
    return K

def k_mixed_partials(X1, Y1, X2, Y2, first_partial_X=True, second_partial_X=True, ls=1., sigma=1.):
    """K_mixed_partials returns mixed partial derivatives of the kernel function with respect to its arguments.
    
    The flags first_partial_X and second_partial_X determine if these derivatives are with respect to X or Y.
    With both set to true, we have
        second_partials[i,j] = \partial^2 / (\partial X1[i] * \partial X2[j] ) K((X1[i], Y1[i]), (X2[j], Y2[j]))
    """
    
    X1.requires_grad_(True)
    Y1.requires_grad_(True)
    X2.requires_grad_(True)
    Y2.requires_grad_(True)

    K = sqr_exp_2D(X1, Y1, X2, Y2, ls=ls, sigma=sigma)
    if first_partial_X:
        if second_partial_X:
            second_partials = torch.autograd.grad(torch.sum(torch.autograd.grad(torch.sum(K), X1, create_graph=True)[0]), X2, create_graph=True)[0]
        else:
            second_partials = torch.autograd.grad(torch.sum(torch.autograd.grad(torch.sum(K), X1, create_graph=True)[0]), Y2, create_graph=True)[0]
    else:
        if second_partial_X:
            second_partials = torch.autograd.grad(torch.sum(torch.autograd.grad(torch.sum(K), Y1, create_graph=True)[0]), X2, create_graph=True)[0]
        else:
            second_partials = torch.autograd.grad(torch.sum(torch.autograd.grad(torch.sum(K), Y1, create_graph=True)[0]), Y2, create_graph=True)[0]
    return second_partials


def sqr_exp_derivative_2D_twodata(XY_1, XY_2, ls, sigma, curl=False):
    """Sqr_exp_derivative_2D computes the covariance matrix of the gradients of a function
    distributed according of a GP with a squared exponential kernel, evaluated at XY.
    Return has shape [2N, 2M]
    Each column of XY_1 is the same, each row of XY_2 is the same.
    Args:
        XY_1: points R^2 at which to compute covariance (torch.Tensor of shape [N, M, 2])
        XY_2: points R^2 at which to compute covariance (torch.Tensor of shape [N, M, 2])
        ls: length scale
        sigma: signal variance
    
    """
    X_1 = XY_1[...,0]
    Y_1 = XY_1[...,1]
    X_2 = XY_2[...,0]
    Y_2 = XY_2[...,1]

    if curl:
        X_1, Y_1 = Y_1, X_1
        X_2, Y_2 = Y_2, X_2
    
    N1 = X_1.shape[0]
    N2 = X_2.shape[1]
    
    Kxx = k_mixed_partials(X_1, Y_1, X_2, Y_2, first_partial_X=True, second_partial_X=True, ls=ls, sigma=sigma)
    Kxy = k_mixed_partials(X_1, Y_1, X_2, Y_2, first_partial_X=True, second_partial_X=False, ls=ls, sigma=sigma)
    Kyx = k_mixed_partials(X_1, Y_1, X_2, Y_2, first_partial_X=False, second_partial_X=True, ls=ls, sigma=sigma)
    Kyy = k_mixed_partials(X_1, Y_1, X_2, Y_2, first_partial_X=False, second_partial_X=False, ls=ls, sigma=sigma)
    
    K_all = stack_matrices(Kxx, Kxy, Kyx, Kyy)
    
    if curl:
        #if curl, I have to negate the derivative wrt xy and yx.
        curl_vec = stack_matrices(torch.ones((N1,N2)), -torch.ones((N1,N2)), -torch.ones((N1,N2)), torch.ones((N1,N2)))
        K_all = curl_vec*K_all
        
    return K_all

def kernel_fcn_helm_twodata(XY_1, XY_2, params):
    
    first_component = sqr_exp_derivative_2D_twodata(XY_1, XY_2, params.ls_1, params.sigma_1) 
    second_component = sqr_exp_derivative_2D_twodata(XY_1, XY_2, params.ls_2, params.sigma_2, curl=True)
    return first_component + second_component

def kernel_fcn_uv_twodata(XY_1, XY_2, params):
    """
    kernel_fcn_uv_twodata computes a covariance kernel for a vector-field evaluated at points XY.
    Args:
        XY_1: points R^2 at which to compute covariance (torch.Tensor of shape [N, M, 2])
        XY_2: points R^2 at which to compute covariance (torch.Tensor of shape [N, M, 2])
        params: parameter of the kernel, from the @dataclass TwoKernelGPParams
    Returns:
        Covariance matrix K (np.array of shape [2*N, 2*M]) where for n in [N],
            K[n,m]=Cov(U[n], U[m]),
            K[N+n,N+m]=Cov(V[n], V[m]), and
            K(n, N+m)=K(N+n, m)=0.
    """
    N = XY_1.shape[0]
    M = XY_1.shape[1]

    X1 = XY_1[...,0]
    Y1 = XY_1[...,1]
    X2 = XY_2[...,0]
    Y2 = XY_2[...,1]

    ker_u = sqr_exp_2D(X1, Y1, X2, Y2, params.ls_1, params.sigma_1) 
    ker_v = sqr_exp_2D(X1, Y1, X2, Y2, params.ls_1, params.sigma_2) 

    kernel = stack_matrices(ker_u, torch.zeros([N,M]), torch.zeros([N,M]), ker_v)
    
    return kernel

def lml(XY_train, UV_train, kind, params):
    """
    lml returns the log marginal likelihood of a GP model over given training data
    Args:
        XY_train: np.array of shape [N_train, 2] with X & Y coordinates of observations
        UV_train: np.array of shape [N_train, 2] with U & V flow observations
        kind: "standard" for velocity GP, "helmholtz" for Helmholtz GP
        params: parameter of the kernel, from the @dataclass TwoKernelGPParams
    """
    if kind=="standard":
        kernel = lambda XY_11, XY_22, params : kernel_fcn_uv_twodata(XY_11, XY_22, params)
    elif kind=="helmholtz":
        kernel = lambda XY_11, XY_22, params : kernel_fcn_helm_twodata(XY_11, XY_22, params)

    train_obs = torch.cat((UV_train[:,0], UV_train[:,1]), dim=0)[:,None]
    M = XY_train.shape[0]
    N = XY_train.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_train[None], [M, 1, 1])
    K_trtr = kernel(XY_11, XY_22, params) + torch.eye(2*M) * params.obs_noise**2
    chol_factor = torch.linalg.cholesky(K_trtr)
    logdet = - torch.sum(torch.log(torch.linalg.diagonal(chol_factor)))
    quadratic_term = -0.5 * torch.sum(torch.square(torch.linalg.solve_triangular(chol_factor, train_obs, upper=False)))
    ll = logdet + quadratic_term
    return ll


def posterior_kernel_twodata(XY_test1, XY_test2, UV_train, XY_train, kind, params):
    """
    Args:
        XY_test1: np.array of shape [N_test, 2] with X & Y coordinates of observations
        XY_test2: np.array of shape [N_test, 2] with X & Y coordinates of observations 
        UV_train: np.array of shape [N_train, 2] with U & V flow observations
        XY_train: np.array of shape [N_train, 2] with X & Y coordinates of test observations, usually on a grid
        kind: flag indicating kernel of interest to be used, can be "standard" or "helmholtz"
        If "standard", for the parameters [1 = u], [2 = v]. If Helmholtz, [1=phi], [2=psi]. 
        params: parameter of the kernel, from the @dataclass TwoKernelGPParams

    Returns:
        posterior mean of size [2*N_test], kernel of size [2*N_test, 2*N_test] for test points, and log marginal likelihood
    """
    
    if kind=="standard":
        kernel = lambda XY_11, XY_22, params : kernel_fcn_uv_twodata(XY_11, XY_22, params)
    elif kind=="helmholtz":
        kernel = lambda XY_11, XY_22, params : kernel_fcn_helm_twodata(XY_11, XY_22, params)
    
    #taking advantage of the new functions (twodata) we compute separately each part in the big prior kernel matrix
    #K_test_test
    M = XY_test1.shape[0]
    N = XY_test2.shape[0]
    XY_11 = torch.tile(XY_test1[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_test2[None], [M, 1, 1])
    K_tete = kernel(XY_11, XY_22, params)

    #K_train_train
    M = XY_train.shape[0]
    N = XY_train.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_train[None], [M, 1, 1])
    K_trtr = kernel(XY_11, XY_22, params) + torch.eye(2*M) * params.obs_noise**2

    #K_test_train
    M = XY_test1.shape[0]
    N = XY_train.shape[0]
    XY_11 = torch.tile(XY_test1[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_train[None], [M, 1, 1])
    K_tetr = kernel(XY_11, XY_22, params)
    
    #K_train_test
    M = XY_train.shape[0]
    N = XY_test2.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_test2[None], [M, 1, 1])
    K_trte = kernel(XY_11, XY_22, params)
    
    assert np.sum((K_tetr - K_trte.T > 1e-4).numpy()) == 0
        
    #compute posterior mean and kernel
    train_obs = torch.cat((UV_train[:,0], UV_train[:,1]), dim=0)[:,None]
    ll = -0.5*torch.matmul(train_obs.T,torch.linalg.solve(K_trtr,train_obs)) - 0.5*torch.linalg.slogdet(K_trtr)[1]
    test_mu = torch.matmul(K_tetr, torch.linalg.solve(K_trtr, train_obs))
    test_cov = K_tete - torch.matmul(K_tetr, torch.linalg.solve(K_trtr, K_trte))
    return test_mu, test_cov, ll
    

def posterior_kernel_twodata_cached(XY_test1, XY_test2, UV_train, XY_train, train_chol, kind, params):
    """
    Args:
        XY_test1: np.array of shape [N_test, 2] with X & Y coordinates of observations
        XY_test2: np.array of shape [N_test, 2] with X & Y coordinates of observations 
        UV_train: np.array of shape [N_train, 2] with U & V flow observations
        XY_train: np.array of shape [N_train, 2] with X & Y coordinates of test observations, usually on a grid
        train_chol: np.array of shape ________ with Cholesky factor of GP prior matrix for training observations 
        kind: flag indicating kernel of interest to be used, can be "standard" or "helmholtz"
        If "standard", for the parameters [1 = u], [2 = v]. If Helmholtz, [1=phi], [2=psi]. 
        params: parameter of the kernel, from the @dataclass TwoKernelGPParams

    Returns:
        posterior mean of size [2*N_test] and kernel of size [2*N_test, 2*N_test] for test points.
    """
    
    if kind=="standard":
        kernel = lambda XY_11, XY_22, params : kernel_fcn_uv_twodata(XY_11, XY_22, params)
    elif kind=="helmholtz":
        kernel = lambda XY_11, XY_22, params : kernel_fcn_helm_twodata(XY_11, XY_22, params)
    #taking advantage of the new functions (twodata) we compute separately each part in the big prior kernel matrix
    #K_test_test
    M = XY_test1.shape[0]
    N = XY_test2.shape[0]
    XY_11 = torch.tile(XY_test1[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_test2[None], [M, 1, 1])
    K_tete = kernel(XY_11, XY_22, params)

    #K_train_test1
    M = XY_train.shape[0]
    N = XY_test1.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_test1[None], [M, 1, 1])
    K_trte1 = kernel(XY_11, XY_22, params)
    
    #K_train_test
    M = XY_train.shape[0]
    N = XY_test2.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_test2[None], [M, 1, 1])
    K_trte2 = kernel(XY_11, XY_22, params)
            
    #compute posterior mean and kernel
    train_obs = torch.cat((UV_train[:,0], UV_train[:,1]), dim=0)[:,None]
    Linv_y = torch.linalg.solve_triangular(train_chol, train_obs, upper=False)
    Linv_cross_top = torch.linalg.solve_triangular(train_chol, K_trte1, upper=False)
    Linv_cross_low = torch.linalg.solve_triangular(train_chol, K_trte2, upper=False)
    posterior_contraction = torch.matmul(Linv_cross_top.t(), Linv_cross_low)
    posterior_mu = torch.matmul(Linv_cross_top.t(), Linv_y)

    posterior_cov = K_tete - posterior_contraction
    return posterior_mu, posterior_cov

def posterior_components_grid(XY_train, UV_train, XY_test1, XY_test2, best_params, curl=False):
    """
    posterior_components_grid computes the posterior mean and variance for either grad phi or curl psi (curl=True)
    on a grid of test points of size GxG := N, given M observations of the current vector field F'. 
    
    Args:
        XY_train: np.array of shape [M, 2] with X & Y coordinates of observations
        UV_train: np.array of shape [M, 2] with U & V flow observations
        XY_test1: np.array of shape [N, 2] with X & Y coordinates of test observations, usually on a grid
        XY_test2: np.array of shape [N, 2] with X & Y coordinates of test observations, usually on a grid
        best_params: list comprising the best parameters obtained through the optimization routine
        curl: if True, compute posterior for the rotation component
        
    """
    
    #First part: prior kernel for the phi or psi component on test points, with shape [2N, 2N]
    M = XY_test1.shape[0]
    N = XY_test2.shape[0]
    XY_11 = torch.tile(XY_test1[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_test2[None], [M, 1, 1])

    if not curl:
        prior_comp = sqr_exp_derivative_2D_twodata(XY_11, XY_22, 
                                            ls = best_params.ls_1, 
                                            sigma = best_params.sigma_1,
                                            curl=False)
    else:
        prior_comp = sqr_exp_derivative_2D_twodata(XY_11, XY_22, 
                                            ls = best_params.ls_2, 
                                            sigma = best_params.sigma_2, 
                                            curl=True)
    
    #Second part: top right [2N x 2M] box, cross covariance between prior phi/psi and training point locations
    M = XY_test1.shape[0]
    N = XY_train.shape[0]
    XY_11 = torch.tile(XY_test1[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_train[None], [M, 1, 1])
    if not curl:
        cross_prior_top = sqr_exp_derivative_2D_twodata(XY_11, XY_22, 
                                           ls = best_params.ls_1, 
                                           sigma = best_params.sigma_1, 
                                           curl=False)
    else:
        cross_prior_top = sqr_exp_derivative_2D_twodata(XY_11, XY_22, 
                                           ls = best_params.ls_2, 
                                           sigma = best_params.sigma_2, 
                                           curl=True)

    #Second part: bottom left [2M x 2N] box, cross covariance between prior phi/psi and training point locations
    M = XY_train.shape[0]
    N = XY_test2.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_test2[None], [M, 1, 1])
    if not curl:
        cross_prior_low = sqr_exp_derivative_2D_twodata(XY_11, XY_22, 
                                           ls = best_params.ls_1, 
                                           sigma = best_params.sigma_1, 
                                           curl=False)
    else:
        cross_prior_low = sqr_exp_derivative_2D_twodata(XY_11, XY_22, 
                                           ls = best_params.ls_2, 
                                           sigma = best_params.sigma_2, 
                                           curl=True)

    #Last part: prior kernel for observations vec_F’ (K_trtr) with K_trtr.shape = [2M, 2M].
    M = XY_train.shape[0]
    N = XY_train.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_train[None], [M, 1, 1])
    prior_obs = kernel_fcn_helm_twodata(XY_11, XY_22, best_params) + torch.eye(2*M)*best_params.obs_noise**2

    #compute posterior mean and kernel
    train_obs = torch.cat((UV_train[:,0], UV_train[:,1]), dim=0)[:,None]
    posterior_mu = torch.matmul(cross_prior_top, torch.linalg.solve(prior_obs, train_obs))
    posterior_cov = prior_comp - torch.matmul(cross_prior_top, torch.linalg.solve(prior_obs , cross_prior_low))
    
    return posterior_mu, posterior_cov

def posterior_components_grid_cached(XY_train, UV_train, XY_test1, XY_test2, train_chol, params, curl=False):
    """
    posterior_components_grid_cached computes the posterior mean and variance for either grad phi or curl psi (curl=True)
    on a grid of test points of size GxG := N, given M observations of the current vector field F', in an efficient way. 
    That is, avoiding the computation of the GP prior for training obs multiple times (by caching the Cholesky factor).
    This is useful for the computation of divergence and vorticity in the Helmholtz GP. 
    
    Args:
        XY_train: np.array of shape [N_train, 2] with X & Y coordinates of test observations, usually on a grid
        UV_train: np.array of shape [N_train, 2] with U & V flow observations
        XY_test1: np.array of shape [N_test, 2] with X & Y coordinates of observations
        XY_test2: np.array of shape [N_test, 2] with X & Y coordinates of observations 
        train_chol: np.array of shape ________ with Cholesky factor of GP prior matrix for training observations 
        params: parameter of the kernel, from the @dataclass TwoKernelGPParams
        curl: if True, compute posterior for the rotation component

    """
    
    #First part: prior kernel for the phi or psi component on test points, with shape [2N, 2N]
    M = XY_test1.shape[0]
    N = XY_test2.shape[0]
    XY_11 = torch.tile(XY_test1[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_test2[None], [M, 1, 1])

    ls, sigma = (params.ls_1, params.sigma_1) if not curl else (params.ls_2, params.sigma_2)
    prior_comp = sqr_exp_derivative_2D_twodata(XY_11, XY_22, ls, sigma, curl)
    
    #Second part: top right [2N x 2M] box, cross covariance between prior phi/psi and training point locations
    M = XY_train.shape[0]
    N = XY_test1.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_test1[None], [M, 1, 1])
    cross_prior_top = sqr_exp_derivative_2D_twodata(XY_11, XY_22, ls, sigma, curl)


    #Second part: bottom left [2M x 2N] box, cross covariance between prior phi/psi and training point locations
    M = XY_train.shape[0]
    N = XY_test2.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_test2[None], [M, 1, 1])
    cross_prior_low = sqr_exp_derivative_2D_twodata(XY_11, XY_22, ls, sigma, curl)

    #compute posterior mean and kernel
    train_obs = torch.cat((UV_train[:,0], UV_train[:,1]), dim=0)[:,None]
    Linv_y = torch.linalg.solve_triangular(train_chol, train_obs, upper=False)
    Linv_cross_top = torch.linalg.solve_triangular(train_chol, cross_prior_top, upper=False)
    Linv_cross_low = torch.linalg.solve_triangular(train_chol, cross_prior_low, upper=False)
    posterior_contraction = torch.matmul(Linv_cross_top.t(), Linv_cross_low)
    posterior_mu = torch.sum(Linv_cross_top * Linv_y, dim=0)[:, None]

    posterior_cov = prior_comp - posterior_contraction

    return posterior_mu, posterior_cov

def posterior_divergence_forloop(XY_test, XY_train, UV_train, kernel, params):
    """
    posterior_divergence_forloop computes the divergence posterior mean and variance at 
    test points for either the velocity GP (kernel='standard') or Helmholtz GP (kernel='helmholtz')
    Args:
        XY_test: torch.tensor of shape [N_test, 2]
        XY_train: torch.tensor of shape [N_train, 2]
        UV_train: torch.tensor of shape [N_train, 2]
        kernel: "helmholtz" or "standard"
        params: parameter of the kernel, from the @dataclass TwoKernelGPParams
    Returns:   
        mean_div: torch.tensor of shape [N_test, 1]
        var_div: torch.tensor of shape [N_test, 1]
    """
    
    mean_div = []
    var_div = []

    # Compute cholesky factor of train matrix 
    M = XY_train.shape[0]
    N = XY_train.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_train[None], [M, 1, 1])
    kernel_fn = kernel_fcn_uv_twodata if kernel=="standard" else kernel_fcn_helm_twodata
    prior_obs = kernel_fn(XY_11, XY_22, params) + torch.eye(2*M)*params.obs_noise**2
    train_chol = torch.linalg.cholesky(prior_obs)
    
    for i in range(XY_test.shape[0]):
        XY_test_i = XY_test[i:i+1,:].clone().requires_grad_(True)
        XY_test_j = XY_test[i:i+1,:].clone().requires_grad_(True)
        # mean_uv_i, cov_uv_i, _ = posterior_kernel_twodata(XY_test_i, XY_test_j, UV_train, XY_train, kernel, params)
        mean_uv_i, cov_uv_i = posterior_kernel_twodata_cached(XY_test_i, XY_test_j, UV_train, XY_train, train_chol, kernel, params) # shapes [2] and [2,2]
        
        mean_div_u_i = torch.autograd.grad(mean_uv_i[0], XY_test_i, create_graph=True)[0][:,0]
        mean_div_v_i = torch.autograd.grad(mean_uv_i[1], XY_test_i, create_graph=True)[0][:,1]
        mean_div_i = mean_div_u_i + mean_div_v_i
        mean_div.append(mean_div_i)

        var_div_uu_i = torch.autograd.grad(torch.autograd.grad(cov_uv_i[0,0], XY_test_i, create_graph=True)[0][:,0], XY_test_j, create_graph=True)[0][:,0]
        var_div_uv_i = torch.autograd.grad(torch.autograd.grad(cov_uv_i[0,1], XY_test_i, create_graph=True)[0][:,0], XY_test_j, create_graph=True)[0][:,1]
        var_div_vu_i = torch.autograd.grad(torch.autograd.grad(cov_uv_i[1,0], XY_test_i, create_graph=True)[0][:,1], XY_test_j, create_graph=True)[0][:,0]
        var_div_vv_i = torch.autograd.grad(torch.autograd.grad(cov_uv_i[1,1], XY_test_i, create_graph=True)[0][:,1], XY_test_j, create_graph=True)[0][:,1]
        
        var_div_i = var_div_uu_i + var_div_uv_i + var_div_vu_i + var_div_vv_i
        var_div.append(var_div_i)
        
    mean_div = torch.tensor(mean_div)[:,None]
    var_div = torch.tensor(var_div)[:,None]

    return mean_div, var_div

def posterior_vorticity_forloop(XY_test, XY_train, UV_train, kernel, params):
    """
    posterior_vorticity_forloop computes the vorticity posterior mean and variance at 
    test points for either the velocity GP (kernel='standard') or Helmholtz GP (kernel='helmholtz')
    Args:
        XY_test: torch.tensor of shape [N_test, 2]
        XY_train: torch.tensor of shape [N_train, 2]
        UV_train: torch.tensor of shape [N_train, 2]
        kernel: "helmholtz" or "standard"
        params: parameter of the kernel, from the @dataclass TwoKernelGPParams
    Returns:   
        mean_vort: torch.tensor of shape [N_test, 1]
        var_vort: torch.tensor of shape [N_test, 1]
    """
    
    mean_vort = []
    var_vort = []

    # Compute cholesky factor of train matrix 
    M = XY_train.shape[0]
    N = XY_train.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_train[None], [M, 1, 1])
    kernel_fn = kernel_fcn_uv_twodata if kernel=="standard" else kernel_fcn_helm_twodata
    prior_obs = kernel_fn(XY_11, XY_22, params) + torch.eye(2*M)*params.obs_noise**2
    train_chol = torch.linalg.cholesky(prior_obs)
    for i in range(XY_test.shape[0]):
        XY_test_i = XY_test[i:i+1,:].clone().requires_grad_(True)
        XY_test_j = XY_test[i:i+1,:].clone().requires_grad_(True)
        
        mean_uv_i, cov_uv_i = posterior_kernel_twodata_cached(XY_test_i, XY_test_j, UV_train, XY_train, train_chol, kernel, params) # shapes [2] and [2,2]
        
        mean_vort_u_i = torch.autograd.grad(mean_uv_i[0], XY_test_i, create_graph=True)[0][:,1]
        mean_vort_v_i = -torch.autograd.grad(mean_uv_i[1], XY_test_i, create_graph=True)[0][:,0]
        mean_vort_i = mean_vort_u_i + mean_vort_v_i
        mean_vort.append(mean_vort_i)

        var_vort_uu_i = torch.autograd.grad(torch.autograd.grad(cov_uv_i[0,0], XY_test_i, create_graph=True)[0][:,1], XY_test_j, create_graph=True)[0][:,1]
        var_vort_uv_i = -torch.autograd.grad(torch.autograd.grad(cov_uv_i[0,1], XY_test_i, create_graph=True)[0][:,1], XY_test_j, create_graph=True)[0][:,0]
        var_vort_vu_i = -torch.autograd.grad(torch.autograd.grad(cov_uv_i[1,0], XY_test_i, create_graph=True)[0][:,0], XY_test_j, create_graph=True)[0][:,1]
        var_vort_vv_i = torch.autograd.grad(torch.autograd.grad(cov_uv_i[1,1], XY_test_i, create_graph=True)[0][:,0], XY_test_j, create_graph=True)[0][:,0]
        
        var_vort_i = var_vort_uu_i + var_vort_uv_i + var_vort_vu_i + var_vort_vv_i
        var_vort.append(var_vort_i)
        
    mean_vort = torch.tensor(mean_vort)[:,None]
    var_vort = torch.tensor(var_vort)[:,None]
    return mean_vort, var_vort

def posterior_divergence_forloop_diffphi(XY_test, XY_train, UV_train, params):
    """
    posterior_divergence_forloop_diffphi computes the divergence posterior mean and variance for 
    the Helmholtz GP in an optimized way (by exploting the fact that the divergence of the vorticity 
    component is 0 and so does not have to be taken into account)
    Args:
        XY_test: torch.tensor of shape [N_test, 2]
        XY_train: torch.tensor of shape [N_train, 2]
        UV_train: torch.tensor of shape [N_train, 2]
        params: parameter of the kernel, from the @dataclass TwoKernelGPParams
    Returns:   
        mean_div: torch.tensor of shape [N_test, 1]
        var_div: torch.tensor of shape [N_test, 1]
    """
    
    mean_div = []
    var_div = []
    M = XY_train.shape[0]
    N = XY_train.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_train[None], [M, 1, 1])
    prior_obs = kernel_fcn_helm_twodata(XY_11, XY_22, params) + torch.eye(2*M)*params.obs_noise**2
    train_chol = torch.linalg.cholesky(prior_obs)
    for i in range(XY_test.shape[0]):
        XY_test_i = XY_test[i:i+1,:].clone().requires_grad_(True)
        XY_test_j = XY_test[i:i+1,:].clone().requires_grad_(True)
        
        mean_uv_i, cov_uv_i = posterior_components_grid_cached(XY_train, UV_train, XY_test_i, XY_test_j, train_chol, params) # shapes [2] and [2,2]
        
        mean_div_u_i = torch.autograd.grad(mean_uv_i[0], XY_test_i, create_graph=True)[0][:,0]
        mean_div_v_i = torch.autograd.grad(mean_uv_i[1], XY_test_i, create_graph=True)[0][:,1]
        mean_div_i = mean_div_u_i + mean_div_v_i
        mean_div.append(mean_div_i)

        var_div_uu_i = torch.autograd.grad(torch.autograd.grad(cov_uv_i[0,0], XY_test_i, create_graph=True)[0][:,0], XY_test_j, create_graph=True)[0][:,0]
        var_div_uv_i = torch.autograd.grad(torch.autograd.grad(cov_uv_i[0,1], XY_test_i, create_graph=True)[0][:,0], XY_test_j, create_graph=True)[0][:,1]
        var_div_vu_i = torch.autograd.grad(torch.autograd.grad(cov_uv_i[1,0], XY_test_i, create_graph=True)[0][:,1], XY_test_j, create_graph=True)[0][:,0]
        var_div_vv_i = torch.autograd.grad(torch.autograd.grad(cov_uv_i[1,1], XY_test_i, create_graph=True)[0][:,1], XY_test_j, create_graph=True)[0][:,1]
        
        var_div_i = var_div_uu_i + var_div_uv_i + var_div_vu_i + var_div_vv_i
        var_div.append(var_div_i)
        
    mean_div = torch.tensor(mean_div)[:,None]
    var_div = torch.tensor(var_div)[:,None]

    return mean_div, var_div
    
def posterior_vorticity_forloop_diffpsi(XY_test, XY_train, UV_train, params):
    """
    posterior_divergence_forloop_diffps computes the vorticity posterior mean and variance for 
    the Helmholtz GP in an optimized way (by exploting the fact that the curl of the divergence 
    component is 0 and so does not have to be taken into account)
    Args:
        XY_test: torch.tensor of shape [N_test, 2]
        XY_train: torch.tensor of shape [N_train, 2]
        UV_train: torch.tensor of shape [N_train, 2]
        params: parameter of the kernel, from the @dataclass TwoKernelGPParams
    Returns:   
        mean_vort: torch.tensor of shape [N_test, 1]
        var_vort: torch.tensor of shape [N_test, 1]
    """
    
    mean_vort = []
    var_vort = []
    M = XY_train.shape[0]
    N = XY_train.shape[0]
    XY_11 = torch.tile(XY_train[:, None], [1, N, 1])
    XY_22 = torch.tile(XY_train[None], [M, 1, 1])
    prior_obs = kernel_fcn_helm_twodata(XY_11, XY_22, params) + torch.eye(2*M)*params.obs_noise**2
    train_chol = torch.linalg.cholesky(prior_obs)
    for i in range(XY_test.shape[0]):
        XY_test_i = XY_test[i:i+1,:].clone().requires_grad_(True)
        XY_test_j = XY_test[i:i+1,:].clone().requires_grad_(True)
        
        mean_uv_i, cov_uv_i = posterior_components_grid_cached(XY_train, UV_train, XY_test_i, XY_test_j, train_chol, params, curl=True) # shapes [2] and [2,2]
        
        mean_vort_u_i = torch.autograd.grad(mean_uv_i[0], XY_test_i, create_graph=True)[0][:,1]
        mean_vort_v_i = -torch.autograd.grad(mean_uv_i[1], XY_test_i, create_graph=True)[0][:,0]
        mean_vort_i = mean_vort_u_i + mean_vort_v_i
        mean_vort.append(mean_vort_i)

        var_vort_uu_i = torch.autograd.grad(torch.autograd.grad(cov_uv_i[0,0], XY_test_i, create_graph=True)[0][:,1], XY_test_j, create_graph=True)[0][:,1]
        var_vort_uv_i = -torch.autograd.grad(torch.autograd.grad(cov_uv_i[0,1], XY_test_i, create_graph=True)[0][:,1], XY_test_j, create_graph=True)[0][:,0]
        var_vort_vu_i = -torch.autograd.grad(torch.autograd.grad(cov_uv_i[1,0], XY_test_i, create_graph=True)[0][:,0], XY_test_j, create_graph=True)[0][:,1]
        var_vort_vv_i = torch.autograd.grad(torch.autograd.grad(cov_uv_i[1,1], XY_test_i, create_graph=True)[0][:,0], XY_test_j, create_graph=True)[0][:,0]
        
        var_vort_i = var_vort_uu_i + var_vort_uv_i + var_vort_vu_i + var_vort_vv_i
        var_vort.append(var_vort_i)
        
    mean_vort = torch.tensor(mean_vort)[:,None]
    var_vort = torch.tensor(var_vort)[:,None]
    return mean_vort, var_vort


def stack_matrices(a, b, c, d):
    # Given conformable matrices a, b, c, d return block matrix [[a, b], [c, d]]
    ab = torch.concatenate([a, b], axis=-1)
    cd = torch.concatenate([c, d], axis=-1)
    ab_cd = torch.concatenate([ab, cd], axis=0) 
    return ab_cd
