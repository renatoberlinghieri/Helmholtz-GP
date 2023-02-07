import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter

# Text and column width for paper submission (pts)
TEXT_WIDTH_PTS = 487.8225
COLUMN_WIDTH_PTS = 234.8775
pts_to_inches_factor = 72
textwidth = TEXT_WIDTH_PTS / pts_to_inches_factor
columnwidth = COLUMN_WIDTH_PTS / pts_to_inches_factor

###############################

def visualize_data(X_grid, Y_grid, XY_train, UV_train, XY_test, UV_test, vorticity, divergence, cmap_div='cool', cmap_vort='plasma', scale=0.03):
    """
    visualize_data produces the plot of the ground truth data: velocity field, divergence, and vorticity.

    Args:
        X_grid: np.array of shape [sqrt(N), sqrt(N)] with X coordinates of test observations on a grid
        Y_grid: np.array of shape [sqrt(N), sqrt(N)] with Y coordinates of test observations on a grid
        XY_train: np.array of shape [M, 2] with X & Y coordinates of observations
        UV_train: np.array of shape [M, 2] with U & V flow observations
        XY_test: np.array of shape [N, 2] with X & Y coordinates of test observations, usually on a grid
        UV_test: np.array of shape [N, 2] with U & V flow ground truth
        vorticity: np.array of shape [sqrt(N), sqrt(N)] with vorticity ground truth (at test points)
        divergence: np.array of shape [sqrt(N), sqrt(N)] with divergence ground truth (at test points)
        cmap_div: color map for the divergence plot
        cmap_vort: color map for the vorticity plot
        scale: scale for plotting the arrows
    """
    f, axarr = plt.subplots(ncols=3, figsize=[18,5])
    
    ax = axarr[0]
    ax.set_title("$F$, ground truth")
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:, 0], UV_train[:, 1], scale_units='xy', scale=scale,label="buoys' locations", color='r')
    ax.quiver(XY_test[:, 0], XY_test[:, 1], UV_test[:, 0], UV_test[:, 1], scale_units='xy', scale=scale, label="ground truth")
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(loc='upper right')

    ax = axarr[1]
    ax.set_title(f"$\delta$, ground truth")
    cs = ax.imshow(divergence, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), cmap=cmap_div, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='black', s=5)
    ax.set_xlabel('Longitude'); #ax.set_ylabel('Latitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    ax = axarr[2]
    ax.set_title(f"$\zeta$, ground truth")
    cs = ax.imshow(vorticity, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='black', s=5)
    ax.set_xlabel('Longitude'); #ax.set_ylabel('Latitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    return


###############################

def plot_results_grid(X_grid, Y_grid, XY_train, UV_train, test_mu, test_cov, levels, scale=0.03, method=""):
    """
    plot_results_grid produces the plot of the velocity prediction and standard deviation.

    Args:
        X_grid: np.array of shape [sqrt(N), sqrt(N)] with X coordinates of test observations on a grid
        Y_grid: np.array of shape [sqrt(N), sqrt(N)] with Y coordinates of test observations on a grid
        XY_train: np.array of shape [M, 2] with X & Y coordinates of observations
        UV_train: np.array of shape [M, 2] with U & V flow observations
        test_mu: np.array of shape [N, 2] with GP posterior mean at test points
        test_cov: np.array of shape [2N, 2N] with GP posterior covariance at test points
        levels: level sets for contour plots
        method: str for adding correct method to plot titles (usually "Helmholtz GP" or "Velocity GP")
    """
    X_grid, Y_grid, XY_train, test_mu, test_cov = _to_numpy([X_grid, Y_grid, XY_train, test_mu, test_cov])
    # format mean and variance for plotting
    test_var = np.diagonal(test_cov)
    grid_points_hor = X_grid.shape[0]
    grid_points_ver = X_grid.shape[1]
    test_mu_grid = np.reshape(test_mu, [2, grid_points_hor, grid_points_ver])
    test_var = np.reshape(test_var, [2, grid_points_hor, grid_points_ver])
    test_std_grid = np.sqrt(test_var[0]) + np.sqrt(test_var[1])

    # Plot the predictive mean and variance conditioned on training points 
    f, axarr = plt.subplots(ncols=2, figsize=[11,5])
    ax = axarr[0]
    ax.set_title(f"$F$, prediction - {method}")
    ax.quiver(X_grid, Y_grid, test_mu_grid[0], test_mu_grid[1], scale_units='xy', scale=scale, label='predicted currents')
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label='buoys locations')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(loc='upper right')

    ax = axarr[1]
    ax.set_title(f"$F$, posterior standard deviation - {method}")
    if levels:
        cs = ax.contourf(X_grid, Y_grid, test_std_grid, levels=levels)
    else:
        cs = ax.contourf(X_grid, Y_grid, test_std_grid)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Longitude');# ax.set_ylabel('Latitude')
    f.colorbar(cs, ax = ax, cmap='inferno')
    ax.legend()

    plt.tight_layout()
    plt.show()
    
    return

###############################

def plot_results_comparison(X_grid, Y_grid, XY_train, UV_train, UV_test, 
                            test_mu_helm, test_cov_helm, 
                            test_mu_std, test_cov_std,
                            scale=0.03, cmap='viridis'):
    """
    plot_results_comparison produces the plot to compare velocity predictions and standard deviation
    between the Helmholtz GP and Velocity GP: first column predictions, second column differences 
    from ground truth, third column standard deviations. 

    Args:
        X_grid: np.array of shape [sqrt(N), sqrt(N)] with X coordinates of test observations on a grid
        Y_grid: np.array of shape [sqrt(N), sqrt(N)] with Y coordinates of test observations on a grid
        XY_train: np.array of shape [M, 2] with X & Y coordinates of observations
        UV_train: np.array of shape [M, 2] with U & V flow observations
        UV_test: np.array of shape [N, 2] with U & V flow ground truth
        test_mu_helm: np.array of shape [N, 2] with Helmholtz GP posterior mean at test points
        test_cov_helm: np.array of shape [2N, 2N] with Helmholtz GP posterior covariance at test points
        test_mu_std: np.array of shape [N, 2] with Velocity GP posterior mean at test points
        test_cov_std: np.array of shape [2N, 2N] with Velocity GP posterior covariance at test points
        scale: scale for plotting arrows
        cmap: color map for standard deviation plots
    """
    grid_points_hor = X_grid.shape[0]
    grid_points_ver = X_grid.shape[1]

    # format mean and variance for plotting
    # helmholtz
    test_var_helm = torch.linalg.diagonal(test_cov_helm)
    test_mu_grid_helm = torch.reshape(test_mu_helm, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_var_helm = torch.reshape(test_var_helm, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_std_grid_helm = np.sqrt(test_var_helm[0]) + np.sqrt(test_var_helm[1])
    # standard
    test_var_std = torch.linalg.diagonal(test_cov_std)
    test_mu_grid_std = torch.reshape(test_mu_std, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_var_std = torch.reshape(test_var_std, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_std_grid_std = np.sqrt(test_var_std[0]) + np.sqrt(test_var_std[1])

    # reshape ground truth for plotting differences
    u_truth = torch.reshape(UV_test[:,0], [grid_points_hor, grid_points_ver]).detach().numpy()
    v_truth = torch.reshape(UV_test[:,1], [grid_points_hor, grid_points_ver]).detach().numpy()
    
    #get max std value for plotting on the same scale
    max_std_helm = np.max(test_std_grid_helm)
    max_std_std = np.max(test_std_grid_std)
    max_val_plot = np.max([max_std_helm, max_std_std])

    # Plot the predictive mean and variance conditioned on training points 
    f, axarr = plt.subplots(nrows=2, ncols=3, figsize=[18,12])

    #top-left
    ax = axarr[0,0]
    ax.set_title("$F$, prediction - Helmholtz GP")
    ax.quiver(X_grid, Y_grid, test_mu_grid_helm[0], test_mu_grid_helm[1], scale_units='xy', scale=scale, label='predicted currents')
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label='buoys locations')
    ax.set_ylabel('Latitude');# ax.set_ylabel('Longitude')
    ax.legend(loc='upper right')

    #top-center
    ax = axarr[0,1]
    ax.set_title("$F$, difference from ground truth - Helmholtz GP")
    ax.quiver(X_grid, Y_grid,  u_truth - test_mu_grid_helm[0], v_truth - test_mu_grid_helm[1], scale_units='xy', scale=scale, label="predicted difference")
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="buoys' locations")
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    ax.legend(loc='upper right')

    #top-right
    ax = axarr[0,2]
    ax.set_title("$F$, posterior standard deviation - Helmholtz GP")
    cs = ax.imshow(test_std_grid_helm, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=0, vmax=max_val_plot, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    ax.legend(loc='upper right')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)
    
    #bottom-left
    ax = axarr[1,0]
    ax.set_title("$F$, prediction - Velocity GP")
    ax.quiver(X_grid, Y_grid, test_mu_grid_std[0], test_mu_grid_std[1], scale_units='xy', scale=scale, label="predicted currents")
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="buoys' locations")
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(loc='upper right')
    

    #bottom-center
    ax = axarr[1,1]
    ax.set_title("$F$, difference from ground truth - Velocity GP")
    ax.quiver(X_grid, Y_grid, u_truth - test_mu_grid_std[0], v_truth - test_mu_grid_std[1], scale_units='xy', scale=scale, label="predicted difference")
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="buoys' locations")
    ax.set_xlabel('Longitude');# ax.set_ylabel('Longitude')
    ax.legend(loc='upper right')

    #bottom-right
    ax = axarr[1,2]
    ax.set_title("$F$, posterior standard deviation - Velocity GP", fontsize=10)
    cs = ax.imshow(test_std_grid_std, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=0, vmax=max_val_plot, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Longitude');# ax.set_ylabel('Longitude')
    ax.legend(loc='upper right')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    plt.tight_layout()
    plt.show()
    
    return

##############################

def plot_results_comparison_realdata(X_grid, Y_grid, XY_train, UV_train,
                                    test_mu_helm, test_cov_helm, 
                                    test_mu_std, test_cov_std,
                                    scale=0.03, cmap='viridis'):
    """
    plot_results_comparison_realdata produces the plot to compare velocity predictions and standard 
    deviation between the Helmholtz GP and Velocity GP for the real data, where we do not have
    ground truth: first column predictions, second column standard deviations. 

    Args:
        X_grid: np.array of shape [sqrt(N), sqrt(N)] with X coordinates of test observations on a grid
        Y_grid: np.array of shape [sqrt(N), sqrt(N)] with Y coordinates of test observations on a grid
        XY_train: np.array of shape [M, 2] with X & Y coordinates of observations
        UV_train: np.array of shape [M, 2] with U & V flow observations
        test_mu_helm: np.array of shape [N, 2] with Helmholtz GP posterior mean at test points
        test_cov_helm: np.array of shape [2N, 2N] with Helmholtz GP posterior covariance at test points
        test_mu_std: np.array of shape [N, 2] with Velocity GP posterior mean at test points
        test_cov_std: np.array of shape [2N, 2N] with Velocity GP posterior covariance at test points
        scale: scale for plotting arrows
        cmap: color map for standard deviation plots
    """

    grid_points_hor = X_grid.shape[0]
    grid_points_ver = X_grid.shape[1]

    # format mean and variance for plotting
    # helmholtz
    test_var_helm = torch.linalg.diagonal(test_cov_helm)
    test_mu_grid_helm = torch.reshape(test_mu_helm, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_var_helm = torch.reshape(test_var_helm, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_std_grid_helm = np.sqrt(test_var_helm[0]) + np.sqrt(test_var_helm[1])
    # standard
    test_var_std = torch.linalg.diagonal(test_cov_std)
    test_mu_grid_std = torch.reshape(test_mu_std, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_var_std = torch.reshape(test_var_std, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_std_grid_std = np.sqrt(test_var_std[0]) + np.sqrt(test_var_std[1])

    #get max std value for plotting on the same scale
    max_std_helm = np.max(test_std_grid_helm)
    max_std_std = np.max(test_std_grid_std)
    max_val_plot = np.max([max_std_helm, max_std_std])

    # Plot the predictive mean and variance conditioned on training points 
    f, axarr = plt.subplots(nrows=2, ncols=2, figsize=[18,12])

    #top-left
    ax = axarr[0,0]
    ax.set_title("$F$, prediction - Helmholtz GP")
    ax.quiver(X_grid, Y_grid, test_mu_grid_helm[0], test_mu_grid_helm[1], scale_units='xy', scale=scale, label='predicted currents')
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label='buoys locations')
    ax.set_ylabel('Latitude');# ax.set_ylabel('Longitude')
    ax.legend(loc='upper right')

    #top-right
    ax = axarr[0,1]
    ax.set_title("$F$, posterior standard deviation - Helmholtz GP")
    cs = ax.imshow(test_std_grid_helm, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=0, vmax=max_val_plot, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    ax.legend(loc='upper right')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)
    
    #bottom-left
    ax = axarr[1,0]
    ax.set_title("$F$, prediction - Velocity GP")
    ax.quiver(X_grid, Y_grid, test_mu_grid_std[0], test_mu_grid_std[1], scale_units='xy', scale=scale, label="predicted currents")
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="buoys' locations")
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(loc='upper right')

    #bottom-right
    ax = axarr[1,1]
    ax.set_title("$F$, posterior standard deviation - Velocity GP")
    cs = ax.imshow(test_std_grid_std, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=0, vmax=max_val_plot, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Longitude');# ax.set_ylabel('Longitude')
    ax.legend(loc='upper right')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    plt.tight_layout()
    plt.show()
    
    return

###############################

def plot_scalarfield_comparison(X_grid, Y_grid, XY_train, div_grid, 
                              mean_div_helm, var_div_helm, mean_div_std, var_div_std, 
                              component="", cmap='cool'):
    """
    plot_scalarfield_comparison produces the plot to compare either the divergence or the vorticity
    predicted by the Helmholtz GP and Velocity GP: first column predictions, second column ground truth,
    third column standard deviations, fourth column z-values. 

    Args:
        X_grid: np.array of shape [sqrt(N), sqrt(N)] with X coordinates of test observations on a grid
        Y_grid: np.array of shape [sqrt(N), sqrt(N)] with Y coordinates of test observations on a grid
        XY_train: np.array of shape [M, 2] with X & Y coordinates of observations
        div_grid: np.array of shape [sqrt(N), sqrt(N)] with divergence (or vorticity) ground truth
        mean_div_helm: np.array of shape [N, 1] with Helmholtz GP posterior mean for divergence/vorticity at test points
        var_div_helm: np.array of shape [N, 1] with Helmholtz GP posterior variance for divergence/vorticity at test points
        mean_div_std: np.array of shape [N, 1] with Velocity GP posterior mean for divergence/vorticity at test points
        var_div_std: np.array of shape [N, 1] with Velocity GP posterior variance for divergence/vorticity at test points
        component: arg to add either "vorticity" or "divergence" to the plot titles
        cmap: color map for imshow plots (usually "cool" for divergence, "plasma" for vorticity)
    """

    grid_points_hor = X_grid.shape[0]
    grid_points_ver = X_grid.shape[1]

    # format mean and variance for plotting
    # helmholtz
    mean_div_helm_grid = torch.reshape(mean_div_helm, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_div_helm_grid = np.sqrt(torch.reshape(var_div_helm, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_helm = (mean_div_helm_grid)/std_div_helm_grid
    # standard
    mean_div_std_grid = torch.reshape(mean_div_std, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_div_std_grid = np.sqrt(torch.reshape(var_div_std, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_std = (mean_div_std_grid)/std_div_std_grid

    # get limit values for plotting on same scales
    # mean
    min_mean_helm, max_mean_helm = np.min(mean_div_helm_grid), np.max(mean_div_helm_grid)
    min_mean_std, max_mean_std = np.min(mean_div_std_grid), np.max(mean_div_std_grid)
    min_mean, max_mean = np.min([min_mean_helm, min_mean_std]), np.max([max_mean_helm, max_mean_std])
    # std
    min_std_helm, max_std_helm = np.min(std_div_helm_grid), np.max(std_div_helm_grid)
    min_std_std, max_std_std = np.min(std_div_std_grid), np.max(std_div_std_grid)
    min_std, max_std = np.min([min_std_helm, min_std_std]), np.max([max_std_helm, max_std_std])
    # z-scores
    min_z_helm, max_z_helm = np.min(z_scores_helm), np.max(z_scores_helm)
    min_z_std, max_z_std = np.min(z_scores_std), np.max(z_scores_std)
    min_z, max_z = np.min([min_z_helm, min_z_std]), np.max([max_z_helm, max_z_std])
    
    # Plot results
    f, axarr = plt.subplots(nrows=2, ncols=4, figsize=[25,12])

    #top-left: mean helmholtz
    ax = axarr[0,0]
    ax.set_title(f"{component}, prediction - Helmholtz GP")
    cs = ax.imshow(mean_div_helm_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Latitude');#ax.set_xlabel('Longitude'); ax.set_ylabel('Longitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #truth helmholtz
    ax = axarr[0,1]
    ax.set_title(f"{component}, ground truth")
    cs = ax.imshow(div_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #top-center: std helmholtz
    ax = axarr[0,2]
    ax.set_title(f"{component}, standard deviation - Helmholtz GP")
    cs = ax.imshow(std_div_helm_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_std, vmax=max_std, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #top-right: z-score helmholtz
    ax = axarr[0,3]
    ax.set_title(f"{component}, $z$-values - Helmholtz GP")
    cs = ax.imshow(z_scores_helm, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_z, vmax=max_z, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #bottom-left: mean std
    ax = axarr[1,0]
    ax.set_title(f"{component}, prediction - Velocity GP")
    cs = ax.imshow(mean_div_std_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #truth standard
    ax = axarr[1,1]
    ax.set_title(f"{component}, ground truth")
    cs = ax.imshow(div_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Longitude');# ax.set_ylabel('Longitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #bottom-center: std std
    ax = axarr[1,2]
    ax.set_title(f"{component}, standard deviation - Velocity GP")
    cs = ax.imshow(std_div_std_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_std, vmax=max_std, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Longitude');# ax.set_ylabel('Longitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #bottom-right: z-score std
    ax = axarr[1,3]
    ax.set_title(f"{component}, $z$-values - Velocity GP")
    cs = ax.imshow(z_scores_std, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_z, vmax=max_z, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Longitude');# ax.set_ylabel('Longitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    plt.tight_layout()
    plt.show()
    
    return

##############################

def plot_scalarfield_comparison_realdata(X_grid, Y_grid, XY_train,
                                        mean_div_helm, var_div_helm, mean_div_std, var_div_std, 
                                        component="", cmap='cool'):
    """
    plot_scalarfield_comparison_realdata produces the plot to compare either the divergence or the vorticity
    predicted by the Helmholtz GP and Velocity GP for the real data, where we do not have ground truth:
    first column predictions, second column standard deviations, third column z-values. 

    Args:
        X_grid: np.array of shape [sqrt(N), sqrt(N)] with X coordinates of test observations on a grid
        Y_grid: np.array of shape [sqrt(N), sqrt(N)] with Y coordinates of test observations on a grid
        XY_train: np.array of shape [M, 2] with X & Y coordinates of observations
        mean_div_helm: np.array of shape [N, 1] with Helmholtz GP posterior mean for divergence/vorticity at test points
        var_div_helm: np.array of shape [N, 1] with Helmholtz GP posterior variance for divergence/vorticity at test points
        mean_div_std: np.array of shape [N, 1] with Velocity GP posterior mean for divergence/vorticity at test points
        var_div_std: np.array of shape [N, 1] with Velocity GP posterior variance for divergence/vorticity at test points
        component: arg to add either "vorticity" or "divergence" to the plot titles
        cmap: color map for imshow plots (usually "cool" for divergence, "plasma" for vorticity)
    """

    grid_points_hor = X_grid.shape[0]
    grid_points_ver = X_grid.shape[1]

    # format mean and variance for plotting
    # helmholtz
    mean_div_helm_grid = torch.reshape(mean_div_helm, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_div_helm_grid = np.sqrt(torch.reshape(var_div_helm, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_helm = (mean_div_helm_grid)/std_div_helm_grid
    # standard
    mean_div_std_grid = torch.reshape(mean_div_std, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_div_std_grid = np.sqrt(torch.reshape(var_div_std, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_std = (mean_div_std_grid)/std_div_std_grid

    # get limit values for plotting on same scales
    # mean
    min_mean_helm, max_mean_helm = np.min(mean_div_helm_grid), np.max(mean_div_helm_grid)
    min_mean_std, max_mean_std = np.min(mean_div_std_grid), np.max(mean_div_std_grid)
    min_mean, max_mean = np.min([min_mean_helm, min_mean_std]), np.max([max_mean_helm, max_mean_std])
    # std
    min_std_helm, max_std_helm = np.min(std_div_helm_grid), np.max(std_div_helm_grid)
    min_std_std, max_std_std = np.min(std_div_std_grid), np.max(std_div_std_grid)
    min_std, max_std = np.min([min_std_helm, min_std_std]), np.max([max_std_helm, max_std_std])
    # z-scores
    min_z_helm, max_z_helm = np.min(z_scores_helm), np.max(z_scores_helm)
    min_z_std, max_z_std = np.min(z_scores_std), np.max(z_scores_std)
    min_z, max_z = np.min([min_z_helm, min_z_std]), np.max([max_z_helm, max_z_std])
    
    # Plot results
    f, axarr = plt.subplots(nrows=2, ncols=3, figsize=[25,12])

    #top-left: mean helmholtz
    ax = axarr[0,0]
    ax.set_title(f"{component}, prediction - Helmholtz GP")
    cs = ax.imshow(mean_div_helm_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Latitude');#ax.set_xlabel('Longitude'); ax.set_ylabel('Longitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #top-center: std helmholtz
    ax = axarr[0,1]
    ax.set_title(f"{component}, standard deviation - Helmholtz GP")
    cs = ax.imshow(std_div_helm_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_std, vmax=max_std, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #top-right: z-score helmholtz
    ax = axarr[0,2]
    ax.set_title(f"{component}, $z$-values - Helmholtz GP")
    cs = ax.imshow(z_scores_helm, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_z, vmax=max_z, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #bottom-left: mean std
    ax = axarr[1,0]
    ax.set_title(f"{component}, prediction - Velocity GP")
    cs = ax.imshow(mean_div_std_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #bottom-center: std std
    ax = axarr[1,1]
    ax.set_title(f"{component}, standard deviation - Velocity GP")
    cs = ax.imshow(std_div_std_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_std, vmax=max_std, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Longitude');# ax.set_ylabel('Longitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #bottom-right: z-score std
    ax = axarr[1,2]
    ax.set_title(f"{component}, $z$-values - Velocity GP")
    cs = ax.imshow(z_scores_std, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_z, vmax=max_z, cmap=cmap)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Longitude');# ax.set_ylabel('Longitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    plt.tight_layout()
    plt.show()
    
    return

###############################

def visualize_dhnn_results(X_grid, Y_grid, XY_train, UV_train, XY_test, UV_pred, vorticity_pred, divergence_pred, cmap_div='cool', cmap_vort='plasma'):
    """
    visualize_dhnn_results produces the plot to visualize the results for the d-hnn approach:
    first column predicted velocity field, second column predicted divergence,
    third column predicted vorticity. 

    Args:
        X_grid: np.array of shape [sqrt(N), sqrt(N)] with X coordinates of test observations on a grid
        Y_grid: np.array of shape [sqrt(N), sqrt(N)] with Y coordinates of test observations on a grid
        XY_train: np.array of shape [M, 2] with X & Y coordinates of observations
        UV_train: np.array of shape [M, 2] with U & V flow observations
        XY_test: np.array of shape [N, 2] with X & Y coordinates of test observations, usually on a grid
        UV_pred: np.array of shape [N, 2] with D-HNN predicted velocity field
        vorticity_pred: np.array of shape [N, 1] with D-HNN predicted vorticity
        divergence_pred: np.array of shape [N, 1] with D-HNN predicted divergence
        cmap_div: color map for imshow plot for divergence
        cmap_vort: color map for imshow plot for vorticity
    """
    grid_points_hor = X_grid.shape[0]
    grid_points_ver = X_grid.shape[1]
    
    #minor preprocessing to get the data in the desired form
    UV_pred = UV_pred.detach()
    vorticity_pred = torch.reshape(vorticity_pred, [grid_points_hor, grid_points_ver]).detach().numpy()
    divergence_pred = torch.reshape(divergence_pred, [grid_points_hor, grid_points_ver]).detach().numpy()
    
    f, axarr = plt.subplots(ncols=3, figsize=[18,5])
    
    ax = axarr[0]
    ax.set_title("$F$, D-HNN prediction")
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:, 0], UV_train[:, 1],label="buoys' locations", color='r')
    ax.quiver(XY_test[:, 0], XY_test[:, 1], UV_pred[:, 0], UV_pred[:, 1],label="predicted current")
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(loc='upper right')

    ax = axarr[1]
    ax.set_title(f"$\delta$, D-HNN prediction")
    cs = ax.imshow(divergence_pred, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), cmap=cmap_div, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='black', s=5)
    ax.set_xlabel('Longitude'); #ax.set_ylabel('Latitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    ax = axarr[2]
    ax.set_title(f"$\zeta$, D-HNN prediction")
    cs = ax.imshow(vorticity_pred, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='black', s=5)
    ax.set_xlabel('Longitude'); #ax.set_ylabel('Latitude')
    ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    return

###############################

def plot_results_paper(
    X_grid,
    Y_grid,
    XY_train,
    UV_train,
    UV_test,
    test_mu_helm,
    test_cov_helm,
    test_mu_std,
    test_cov_std,
    div_grid,
    mean_div_helm,
    var_div_helm,
    mean_div_std,
    var_div_std,
    cmap="cool",
    scale=2,
    data_scale=0.5,
):
    """
    plot_results_paper produces the plot that we include in the main body of the paper to compare
    velocity and divergence predictions.

    Args:
        X_grid: np.array of shape [sqrt(N), sqrt(N)] with X coordinates of test observations on a grid
        Y_grid: np.array of shape [sqrt(N), sqrt(N)] with Y coordinates of test observations on a grid
        XY_train: np.array of shape [M, 2] with X & Y coordinates of observations
        UV_train: np.array of shape [M, 2] with U & V flow observations
        UV_test: np.array of shape [N, 2] with U & V flow ground truth
        test_mu_helm: np.array of shape [N, 2] with Helmholtz GP posterior mean at test points
        test_cov_helm: np.array of shape [2N, 2N] with Helmholtz GP posterior covariance at test points
        test_mu_std: np.array of shape [N, 2] with Velocity GP posterior mean at test points
        test_cov_std: np.array of shape [2N, 2N] with Velocity GP posterior covariance at test points
        div_grid: np.array of shape [sqrt(N), sqrt(N)] with divergence ground truth
        mean_div_helm: np.array of shape [N, 1] with Helmholtz GP posterior mean for divergence at test points
        var_div_helm: np.array of shape [N, 1] with Helmholtz GP posterior variance for divergence at test points
        mean_div_std: np.array of shape [N, 1] with Velocity GP posterior mean for divergence at test points
        var_div_std: np.array of shape [N, 1] with Velocity GP posterior variance for divergence at test points
        cmap: color map for imshow plots 
        scale: scale for plotting the arrows
        data_scale: scale for plotting the observation (dots)
    """


    grid_points_hor = X_grid.shape[0]
    grid_points_ver = X_grid.shape[1]

    # format prediction mean and variance for plotting
    # helmholtz
    test_var_helm = torch.linalg.diagonal(test_cov_helm)
    test_mu_grid_helm = torch.reshape(test_mu_helm, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_var_helm = torch.reshape(test_var_helm, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_std_grid_helm = np.sqrt(test_var_helm[0]) + np.sqrt(test_var_helm[1])
    # standard
    test_var_std = torch.linalg.diagonal(test_cov_std)
    test_mu_grid_std = torch.reshape(test_mu_std, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_var_std = torch.reshape(test_var_std, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_std_grid_std = np.sqrt(test_var_std[0]) + np.sqrt(test_var_std[1])

    # reshape ground truth for plotting it
    u_truth = torch.reshape(UV_test[:,0], [grid_points_hor, grid_points_ver]).detach().numpy()
    v_truth = torch.reshape(UV_test[:,1], [grid_points_hor, grid_points_ver]).detach().numpy()
    
    # format mean and variance for plotting
    # helmholtz
    mean_div_helm_grid = torch.reshape(mean_div_helm, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_div_helm_grid = np.sqrt(torch.reshape(var_div_helm, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_helm = (mean_div_helm_grid)/std_div_helm_grid
    # standard
    mean_div_std_grid = torch.reshape(mean_div_std, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_div_std_grid = np.sqrt(torch.reshape(var_div_std, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_std = (mean_div_std_grid)/std_div_std_grid

    # get limit values for plotting on same scales
    # mean
    min_mean_helm, max_mean_helm = np.min(mean_div_helm_grid), np.max(mean_div_helm_grid)
    min_mean_std, max_mean_std = np.min(mean_div_std_grid), np.max(mean_div_std_grid)
    min_mean, max_mean = np.min([min_mean_helm, min_mean_std]), np.max([max_mean_helm, max_mean_std])
    # std
    min_std_helm, max_std_helm = np.min(std_div_helm_grid), np.max(std_div_helm_grid)
    min_std_std, max_std_std = np.min(std_div_std_grid), np.max(std_div_std_grid)
    min_std, max_std = np.min([min_std_helm, min_std_std]), np.max([max_std_helm, max_std_std])
    # z-scores
    min_z_helm, max_z_helm = np.min(z_scores_helm), np.max(z_scores_helm)
    min_z_std, max_z_std = np.min(z_scores_std), np.max(z_scores_std)
    min_z, max_z = np.min([min_z_helm, min_z_std]), np.max([max_z_helm, max_z_std])

    # Plot results
    f, axarr = plt.subplots(
        nrows=2,
        ncols=4,
        sharey="row",
        sharex="col",
        figsize=(textwidth, 0.4 * textwidth),
    )

    # top-left: ground truth
    ax = axarr[0, 0]
    ax.set_title(f"$F$, ground truth", y=0.95)
    ax.quiver(X_grid, Y_grid, u_truth, v_truth, scale_units="xy", scale=scale)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    ax.quiver(
        XY_train[:, 0],
        XY_train[:, 1],
        UV_train[:, 0],
        UV_train[:, 1],
        color="red",
        scale_units="xy",
        scale=scale,
    )
    ax.set_ylabel("Latitude")

    # top-left-center: prediction helmholtz
    ax = axarr[0, 1]
    ax.set_title("$F$, Helmholtz GP", y=0.95)
    ax.quiver(
        X_grid,
        Y_grid,
        test_mu_grid_helm[0],
        test_mu_grid_helm[1],
        scale_units="xy",
        scale=scale,
    )
    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    ax.quiver(
        XY_train[:, 0],
        XY_train[:, 1],
        UV_train[:, 0],
        UV_train[:, 1],
        color="red",
        scale_units="xy",
        scale=scale,
    )

    # top-right-center: divergence helmholtz
    ax = axarr[0, 2]
    ax.set_title(f"$\delta$, Helmholtz GP", y=0.95)
    cs = ax.imshow(
        mean_div_helm_grid,
        origin="lower",
        extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)),
        vmin=min_mean,
        vmax=max_mean,
        cmap=cmap,
        aspect="auto",
    )
    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    add_colorbar(ax, cs)

    # top-right: z-score helmholtz
    ax = axarr[0, 3]
    ax.set_title("$\delta$ ($z$-value), Helmholtz GP", y=0.95)
    cs = ax.imshow(
        z_scores_helm,
        origin="lower",
        extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)),
        vmin=min_z,
        vmax=max_z,
        cmap=cmap,
        aspect="auto",
    )
    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    add_colorbar(ax, cs)

    # bottom-left: ground truth divergence
    ax = axarr[1, 0]
    ax.set_title(f"$\delta$, ground truth", y=0.95)
    cs = ax.imshow(
        div_grid,
        origin="lower",
        extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)),
        vmin=min_mean,
        vmax=max_mean,
        cmap=cmap,
        aspect="auto",
    )
    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    ax.set_ylabel("Latitude")
    ax.set_xlabel("Longitude")
    add_colorbar(ax, cs)

    # bottom-left-center: prediction standard
    ax = axarr[1, 1]
    ax.set_title("$F$, Velocity GP", y=0.95)
    ax.quiver(
        X_grid,
        Y_grid,
        test_mu_grid_std[0],
        test_mu_grid_std[1],
        scale_units="xy",
        scale=scale,
        label="Current",
    )

    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    ax.quiver(
        XY_train[:, 0],
        XY_train[:, 1],
        UV_train[:, 0],
        UV_train[:, 1],
        color="red",
        scale_units="xy",
        scale=scale,
        label="Buoy",
    )
    ax.set_xlabel("Longitude")

    # bottom-right-center: predicted divergence standard
    ax = axarr[1, 2]
    ax.set_title("$\delta$, Velocity GP", y=0.95)
    cs = ax.imshow(
        mean_div_std_grid,
        origin="lower",
        extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)),
        vmin=min_mean,
        vmax=max_mean,
        cmap=cmap,
        aspect="auto",
    )
    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    ax.set_xlabel("Longitude")

    add_colorbar(ax, cs)

    # bottom-right: z-score std
    ax = axarr[1, 3]
    ax.set_title("$\delta$ ($z$-value), Velocity GP", y=0.95)
    cs = ax.imshow(
        z_scores_std,
        origin="lower",
        extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)),
        vmin=min_z,
        vmax=max_z,
        cmap=cmap,
        aspect="auto",
    )
    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    ax.set_xlabel("Longitude")

    add_colorbar(ax, cs)
    remove_ticks(np.reshape(axarr, -1))
    handles, labels = axarr[1, 1].get_legend_handles_labels()
    legend = f.legend(
        labels=labels,
        handles=handles,
        bbox_to_anchor=[0.5, 0.02],
        loc="center",
        ncol=2,
        frameon=False,
        markerscale=3.0,
    )

    plt.tight_layout()
    plt.show()
    
    return

#################################

def plot_results_paper_realdata(
    X_grid,
    Y_grid,
    XY_train,
    UV_train,
    test_mu_helm,
    test_cov_helm,
    test_mu_std,
    test_cov_std,
    mean_div_helm,
    var_div_helm,
    mean_div_std,
    var_div_std,
    cmap="cool",
    scale=2,
    data_scale=0.25,
    buoy_width=0.003,
):
    """
    plot_results_paper_realdata produces the plot that we include in the main body of the paper to compare
    velocity and divergence predictions for real data tasks (where we do not have ground truth)
    
    Args:
        X_grid: np.array of shape [sqrt(N), sqrt(N)] with X coordinates of test observations on a grid
        Y_grid: np.array of shape [sqrt(N), sqrt(N)] with Y coordinates of test observations on a grid
        XY_train: np.array of shape [M, 2] with X & Y coordinates of observations
        UV_train: np.array of shape [M, 2] with U & V flow observations
        test_mu_helm: np.array of shape [N, 2] with Helmholtz GP posterior mean at test points
        test_cov_helm: np.array of shape [2N, 2N] with Helmholtz GP posterior covariance at test points
        test_mu_std: np.array of shape [N, 2] with Velocity GP posterior mean at test points
        test_cov_std: np.array of shape [2N, 2N] with Velocity GP posterior covariance at test points
        mean_div_helm: np.array of shape [N, 1] with Helmholtz GP posterior mean for divergence at test points
        var_div_helm: np.array of shape [N, 1] with Helmholtz GP posterior variance for divergence at test points
        mean_div_std: np.array of shape [N, 1] with Velocity GP posterior mean for divergence at test points
        var_div_std: np.array of shape [N, 1] with Velocity GP posterior variance for divergence at test points
        cmap: color map for imshow plots 
        scale: scale for plotting the arrows
        data_scale: scale for plotting the observation (dots)
        buoy_width: width of arrow for training points
    """

    grid_points_hor = X_grid.shape[0]
    grid_points_ver = X_grid.shape[1]

    # format prediction mean and variance for plotting
    # helmholtz
    test_var_helm = np.diagonal(test_cov_helm.detach().numpy())
    test_mu_grid_helm = np.reshape(test_mu_helm.detach().numpy(), [2, grid_points_hor, grid_points_ver])
    test_var_helm = np.reshape(test_var_helm, [2, grid_points_hor, grid_points_ver])
    test_std_grid_helm = np.sqrt(test_var_helm[0]) + np.sqrt(test_var_helm[1])
    # standard
    test_var_std = np.diagonal(test_cov_std.detach().numpy())
    test_mu_grid_std = np.reshape(test_mu_std.detach().numpy(), [2, grid_points_hor, grid_points_ver])
    test_var_std = np.reshape(test_var_std, [2, grid_points_hor, grid_points_ver])
    test_std_grid_std = np.sqrt(test_var_std[0]) + np.sqrt(test_var_std[1])

    # format mean and variance for plotting
    # helmholtz
    mean_div_helm_grid = np.reshape(mean_div_helm.detach().numpy(), [grid_points_hor, grid_points_ver])
    std_div_helm_grid = np.sqrt(
        np.reshape(var_div_helm.detach().numpy(), [grid_points_hor, grid_points_ver])
    )
    z_scores_helm = (mean_div_helm_grid) / std_div_helm_grid
    # standard
    mean_div_std_grid = np.reshape(mean_div_std.detach().numpy(), [grid_points_hor, grid_points_ver])
    std_div_std_grid = np.sqrt(
        np.reshape(var_div_std.detach().numpy(), [grid_points_hor, grid_points_ver])
    )
    z_scores_std = (mean_div_std_grid) / std_div_std_grid

    # get limit values for plotting on same scales
    # mean
    min_mean_helm, max_mean_helm = np.min(mean_div_helm_grid), np.max(
        mean_div_helm_grid
    )
    min_mean_std, max_mean_std = np.min(mean_div_std_grid), np.max(mean_div_std_grid)
    min_mean, max_mean = np.min([min_mean_helm, min_mean_std]), np.max(
        [max_mean_helm, max_mean_std]
    )
    # std
    min_std_helm, max_std_helm = np.min(std_div_helm_grid), np.max(std_div_helm_grid)
    min_std_std, max_std_std = np.min(std_div_std_grid), np.max(std_div_std_grid)
    min_std, max_std = np.min([min_std_helm, min_std_std]), np.max(
        [max_std_helm, max_std_std]
    )
    # z-scores
    min_z_helm, max_z_helm = np.min(z_scores_helm), np.max(z_scores_helm)
    min_z_std, max_z_std = np.min(z_scores_std), np.max(z_scores_std)
    min_z, max_z = np.min([np.min([min_z_helm, min_z_std]), -1]), np.max(
        [np.max([max_z_helm, max_z_std]), 1]
    )

    # Plot results
    f, axarr = plt.subplots(
        nrows=2,
        ncols=3,
        sharey="row",
        sharex="col",
        figsize=(textwidth, 0.4 * textwidth),
    )

    # top-left-center: prediction helmholtz
    ax = axarr[0, 0]
    ax.set_title("$F$, Helmholtz GP", y=0.95)
    ax.quiver(
        X_grid,
        Y_grid,
        test_mu_grid_helm[0],
        test_mu_grid_helm[1],
        scale_units="xy",
        scale=scale,
    )
    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    ax.quiver(
        XY_train[:, 0],
        XY_train[:, 1],
        UV_train[:, 0],
        UV_train[:, 1],
        color="red",
        scale_units="xy",
        scale=scale,
        width=buoy_width
    )
    ax.set_ylabel("Latitude")

    # top-right-center: divergence helmholtz
    ax = axarr[0, 1]
    ax.set_title(f"$\delta$, Helmholtz GP", y=0.95)
    cs = ax.imshow(
        mean_div_helm_grid,
        origin="lower",
        extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)),
        vmin=min_mean,
        vmax=max_mean,
        cmap=cmap,
        aspect="auto",
    )
    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    add_colorbar(ax, cs)

    # top-right: z-score helmholtz
    ax = axarr[0, 2]
    ax.set_title("$\delta$ ($z$-value), Helmholtz GP", y=0.95)
    cs = ax.imshow(
        z_scores_helm,
        origin="lower",
        extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)),
        vmin=min_z,
        vmax=max_z,
        cmap=cmap,
        aspect="auto",
    )
    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    add_colorbar(ax, cs)

    # bottom-left-center: prediction standard
    ax = axarr[1, 0]
    ax.set_title("$F$, Velocity GP", y=0.95)
    ax.quiver(
        X_grid,
        Y_grid,
        test_mu_grid_std[0],
        test_mu_grid_std[1],
        scale_units="xy",
        scale=scale,
        label="Current",
    )

    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    ax.quiver(
        XY_train[:, 0],
        XY_train[:, 1],
        UV_train[:, 0],
        UV_train[:, 1],
        color="red",
        scale_units="xy",
        scale=scale,
        label="Buoy",
        width=buoy_width
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # bottom-right-center: predicted divergence standard
    ax = axarr[1, 1]
    ax.set_title("$\delta$, Velocity GP", y=0.95)
    cs = ax.imshow(
        mean_div_std_grid,
        origin="lower",
        extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)),
        vmin=min_mean,
        vmax=max_mean,
        cmap=cmap,
        aspect="auto",
    )
    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    ax.set_xlabel("Longitude")

    add_colorbar(ax, cs)

    # bottom-right: z-score std
    ax = axarr[1, 2]
    ax.set_title("$\delta$ ($z$-value), Velocity GP", y=0.95)
    cs = ax.imshow(
        z_scores_std,
        origin="lower",
        extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)),
        vmin=min_z,
        vmax=max_z,
        cmap=cmap,
        aspect="auto",
    )
    ax.scatter(XY_train[:, 0], XY_train[:, 1], c="r", s=data_scale)
    ax.set_xlabel("Longitude")

    add_colorbar(ax, cs)
    remove_ticks(np.reshape(axarr, -1))
    handles, labels = axarr[1, 1].get_legend_handles_labels()
    legend = f.legend(
        labels=labels,
        handles=handles,
        bbox_to_anchor=[0.5, 0.001],
        loc="center",
        ncol=2,
        frameon=False,
        markerscale=3.0,
    )

    plt.tight_layout()
    plt.show()

    return

############

def plot_results_appendix(X_grid, Y_grid, XY_train, UV_train, UV_test, 
                        test_mu_helm, test_cov_helm, 
                        test_mu_std, test_cov_std,
                        div_grid, 
                        mean_div_helm, var_div_helm, 
                        mean_div_std, var_div_std, 
                        vort_grid, 
                        mean_vort_helm, var_vort_helm, 
                        mean_vort_std, var_vort_std,
                        pred_dhnn, 
                        div_dhnn, vort_dhnn, 
                        cmap='cool', cmap_vort='plasma', scale=3,
                        save_dest = ""):
    """
    plot_results_appendix produces the plot that we include in the appendix of the paper to compare
    velocity, divergence and vorticity predictions, for Helmholtz GP, velocity GP, and D-HNN.

    Args:
        X_grid: np.array of shape [sqrt(N), sqrt(N)] with X coordinates of test observations on a grid
        Y_grid: np.array of shape [sqrt(N), sqrt(N)] with Y coordinates of test observations on a grid
        XY_train: np.array of shape [M, 2] with X & Y coordinates of observations
        UV_train: np.array of shape [M, 2] with U & V flow observations
        UV_test: np.array of shape [N, 2] with U & V flow ground truth
        test_mu_helm: np.array of shape [N, 2] with Helmholtz GP posterior mean at test points
        test_cov_helm: np.array of shape [2N, 2N] with Helmholtz GP posterior covariance at test points
        test_mu_std: np.array of shape [N, 2] with Velocity GP posterior mean at test points
        test_cov_std: np.array of shape [2N, 2N] with Velocity GP posterior covariance at test points
        div_grid: np.array of shape [sqrt(N), sqrt(N)] with divergence ground truth
        mean_div_helm: np.array of shape [N, 1] with Helmholtz GP posterior mean for divergence at test points
        var_div_helm: np.array of shape [N, 1] with Helmholtz GP posterior variance for divergence at test points
        mean_div_std: np.array of shape [N, 1] with Velocity GP posterior mean for divergence at test points
        var_div_std: np.array of shape [N, 1] with Velocity GP posterior variance for divergence at test points
        vort_grid: np.array of shape [sqrt(N), sqrt(N)] with vorticity ground truth
        mean_vort_helm: np.array of shape [N, 1] with Helmholtz GP posterior mean for vorticity at test points
        var_vort_helm: np.array of shape [N, 1] with Helmholtz GP posterior variance for vorticity at test points
        mean_vort_std: np.array of shape [N, 1] with Velocity GP posterior mean for vorticity at test points
        var_vort_std: np.array of shape [N, 1] with Velocity GP posterior variance for vorticity at test points
        pred_dhnn: np.array of shape [N, 2] with D-HNN predicted velocity field
        div_dhnn: np.array of shape [N, 1] with D-HNN predicted divergence
        vort_dhnn: np.array of shape [N, 1] with D-HNN predicted vorticity
        cmap: color map for divergence plots
        cmap_vort: color map for vorticity plots
        scale: scale for plotting the arrows
        save_dest: path where you want your file to be saved in .pdf format
    """
    grid_points_hor = X_grid.shape[0]
    grid_points_ver = X_grid.shape[1]

    # format prediction mean and variance for plotting
    # helmholtz
    test_var_helm = torch.linalg.diagonal(test_cov_helm)
    test_mu_grid_helm = torch.reshape(test_mu_helm, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_var_helm = torch.reshape(test_var_helm, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_std_grid_helm = np.sqrt(test_var_helm[0]) + np.sqrt(test_var_helm[1])
    # standard
    test_var_std = torch.linalg.diagonal(test_cov_std)
    test_mu_grid_std = torch.reshape(test_mu_std, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_var_std = torch.reshape(test_var_std, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_std_grid_std = np.sqrt(test_var_std[0]) + np.sqrt(test_var_std[1])

    # reshape ground truth for plotting it
    u_truth = torch.reshape(UV_test[:,0], [grid_points_hor, grid_points_ver]).detach().numpy()
    v_truth = torch.reshape(UV_test[:,1], [grid_points_hor, grid_points_ver]).detach().numpy()

    #minor preprocessing to get the data in the desired form
    pred_dhnn = pred_dhnn.detach().numpy()
    vort_dhnn = torch.reshape(vort_dhnn, [grid_points_hor, grid_points_ver]).detach().numpy()
    div_dhnn = torch.reshape(div_dhnn, [grid_points_hor, grid_points_ver]).detach().numpy()
    
    #reshape d-hnn predictions
    u_dhnn = pred_dhnn[:,0].reshape(X_grid.shape)
    v_dhnn =pred_dhnn[:,1].reshape(X_grid.shape)

    # format mean and variance divergence for plotting
    # helmholtz
    mean_div_helm_grid = torch.reshape(mean_div_helm, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_div_helm_grid = np.sqrt(torch.reshape(var_div_helm, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_helm = (mean_div_helm_grid)/std_div_helm_grid
    # standard
    mean_div_std_grid = torch.reshape(mean_div_std, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_div_std_grid = np.sqrt(torch.reshape(var_div_std, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_std = (mean_div_std_grid)/std_div_std_grid

    # get limit values for divergence plotting on same scales
    # mean
    min_mean_helm, max_mean_helm = np.min(mean_div_helm_grid), np.max(mean_div_helm_grid)
    min_mean_std, max_mean_std = np.min(mean_div_std_grid), np.max(mean_div_std_grid)
    min_mean, max_mean = np.min([min_mean_helm, min_mean_std]), np.max([max_mean_helm, max_mean_std])
    # std
    min_std_helm, max_std_helm = np.min(std_div_helm_grid), np.max(std_div_helm_grid)
    min_std_std, max_std_std = np.min(std_div_std_grid), np.max(std_div_std_grid)
    min_std, max_std = np.min([min_std_helm, min_std_std]), np.max([max_std_helm, max_std_std])
    # z-scores
    min_z_helm, max_z_helm = np.min(z_scores_helm), np.max(z_scores_helm)
    min_z_std, max_z_std = np.min(z_scores_std), np.max(z_scores_std)
    min_z, max_z = np.min([min_z_helm, min_z_std]), np.max([max_z_helm, max_z_std])

    # format mean and variance vorticity for plotting
    # helmholtz
    mean_vort_helm_grid = torch.reshape(mean_vort_helm, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_vort_helm_grid = np.sqrt(torch.reshape(var_vort_helm, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_helm_vort = (mean_vort_helm_grid)/std_vort_helm_grid
    # standard
    mean_vort_std_grid = torch.reshape(mean_vort_std, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_vort_std_grid = np.sqrt(torch.reshape(var_vort_std, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_std_vort = (mean_vort_std_grid)/std_vort_std_grid

    # get limit values for divergence plotting on same scales
    # mean
    min_mean_helm_vort, max_mean_helm_vort = np.min(mean_vort_helm_grid), np.max(mean_vort_helm_grid)
    min_mean_std_vort, max_mean_std_vort = np.min(mean_vort_std_grid), np.max(mean_vort_std_grid)
    min_mean_vort, max_mean_vort = np.min([min_mean_helm_vort, min_mean_std_vort]), np.max([max_mean_helm_vort, max_mean_std_vort])
    # std
    min_std_helm_vort, max_std_helm_vort = np.min(std_vort_helm_grid), np.max(std_vort_helm_grid)
    min_std_std_vort, max_std_std_vort = np.min(std_vort_std_grid), np.max(std_vort_std_grid)
    min_std_vort, max_std_vort = np.min([min_std_helm_vort, min_std_std_vort]), np.max([max_std_helm_vort, max_std_std_vort])
    # z-scores
    min_z_helm_vort, max_z_helm_vort = np.min(z_scores_helm_vort), np.max(z_scores_helm_vort)
    min_z_std_vort, max_z_std_vort = np.min(z_scores_std_vort), np.max(z_scores_std_vort)
    min_z_vort, max_z_vort = np.min([min_z_helm_vort, min_z_std_vort]), np.max([max_z_helm_vort, max_z_std_vort])

    ##############################

    f, axarr = plt.subplots(nrows=8, ncols=4, figsize=[textwidth,1.28*textwidth], sharex='row', sharey='col')

    #1-1: ground truth
    ax = axarr[0,0]
    ax.set_title(f"$F$, ground truth")
    ax.quiver(X_grid, Y_grid, u_truth, v_truth, scale_units='xy', scale=scale, label="Current")
    #ax.scatter(XY_train[:, 0], XY_train[:, 1], c='r', s=2)
    #ax.scatter(X_grid, Y_grid, c='black', s=1)
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="Buoy")
    ax.set_ylabel('Latitude');
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc='upper right')

    #1-2: prediction helmholtz
    ax = axarr[0,1]
    ax.set_title("$F$, Helmholtz GP")
    ax.quiver(X_grid, Y_grid, test_mu_grid_helm[0], test_mu_grid_helm[1], scale_units='xy', scale=scale, label="Current")
    #ax.scatter(XY_train[:, 0], XY_train[:, 1], c='r', s=2)
    #ax.scatter(X_grid, Y_grid, c='black', s=1)
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="Buoy")
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc='upper right')

    #1-3: prediction standard
    ax = axarr[0,2]
    ax.set_title("$F$, Velocity GP")
    ax.quiver(X_grid, Y_grid, test_mu_grid_std[0], test_mu_grid_std[1], scale_units='xy', scale=scale, label="Current")
    #ax.scatter(XY_train[:, 0], XY_train[:, 1], c='r', s=2)
    #ax.scatter(X_grid, Y_grid, c='black', s=1)
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="Buoy")
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc='upper right')

    #1-4: prediction dhnn
    ax = axarr[0,3]
    ax.set_title("$F$, D-HNN")
    ax.quiver(X_grid, Y_grid, u_dhnn, v_dhnn, scale_units='xy', scale=scale, label="Current")
    #ax.scatter(XY_train[:, 0], XY_train[:, 1], c='r', s=2)
    #ax.scatter(X_grid, Y_grid, c='black', s=1)
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="Buoy")
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc='upper right')

    ##############################

    #2-1: empty
    ax = axarr[1,0]
    ax.set_ylabel('Latitude');
    
    #2-2: difference prediction helmholtz
    ax = axarr[1,1]
    ax.set_title("Diff from truth, Helmholtz GP")
    ax.quiver(X_grid, Y_grid, u_truth - test_mu_grid_helm[0], v_truth - test_mu_grid_helm[1], scale_units='xy', scale=scale, label="Current",)
    #ax.scatter(XY_train[:, 0], XY_train[:, 1], c='r', s=2)
    #ax.scatter(X_grid, Y_grid, c='black', s=1)
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="Buoy")
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc='upper right')

    #2-3: difference prediction standard
    ax = axarr[1,2]
    ax.set_title("Diff from truth, Velocity GP")
    ax.quiver(X_grid, Y_grid, u_truth - test_mu_grid_std[0], v_truth - test_mu_grid_std[1], scale_units='xy', scale=scale, label="Current",)
    #ax.scatter(XY_train[:, 0], XY_train[:, 1], c='r', s=2)
    #ax.scatter(X_grid, Y_grid, c='black', s=1)
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="Buoy")
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc='upper right')

    #2-4: difference prediction dhnn
    ax = axarr[1,3]
    ax.set_title("Diff from truth, D-HNN")
    ax.quiver(X_grid, Y_grid, u_truth - u_dhnn, v_truth - v_dhnn, scale_units='xy', scale=scale, label="Current",)
    #ax.scatter(XY_train[:, 0], XY_train[:, 1], c='r', s=2)
    #ax.scatter(X_grid, Y_grid, c='black', s=1)
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="Buoy",)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc='upper right')

    ##############################

    #3-1: ground truth divergence
    ax = axarr[2,0]
    ax.set_title(f"$\delta$, ground truth")
    cs = ax.imshow(div_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_ylabel('Latitude');
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #3-2: divergence helmholtz
    ax = axarr[2,1]
    ax.set_title(f"$\delta$, Helmholtz GP")
    cs = ax.imshow(mean_div_helm_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #3-3: divergence standard
    ax = axarr[2,2]
    ax.set_title(f"$\delta$, Velocity GP")
    cs = ax.imshow(mean_div_std_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #3-4: divergence dhnn
    ax = axarr[2,3]
    ax.set_title(f"$\delta$, D-HNN")
    cs = ax.imshow(div_dhnn, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    ##############################

    #4-1: empty
    ax = axarr[3,0]
    ax.set_ylabel('Latitude');

    #4-2: divergence std helmholtz
    ax = axarr[3,1]
    ax.set_title(f"$\delta$ (std), Helmholtz GP")
    cs = ax.imshow(std_div_helm_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_std, vmax=max_std, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #4-3: divergence std standard
    ax = axarr[3,2]
    ax.set_title(f"$\delta$ (std), Velocity GP")
    cs = ax.imshow(std_div_std_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_std, vmax=max_std, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #4-4: empty
    ax = axarr[3,3]
   
   ##############################

    #5-1: empty
    ax = axarr[4,0]
    ax.set_ylabel('Latitude');

    #5-2: z-score helmholtz
    ax = axarr[4,1]
    ax.set_title("$\delta$ ($z$-value), Helmholtz GP")
    cs = ax.imshow(z_scores_helm, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_z, vmax=max_z, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #5-3: z-score std
    ax = axarr[4,2]
    ax.set_title("$\delta$ ($z$-value), Velocity GP")
    cs = ax.imshow(z_scores_std, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_z, vmax=max_z, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #5-4: empty
    ax = axarr[4,3]

    ##############################

    #6-1: ground truth vorticity
    ax = axarr[5,0]
    ax.set_title(f"$\zeta$, ground truth")
    cs = ax.imshow(vort_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean_vort, vmax=max_mean_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_ylabel('Latitude');
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #6-2: vorticity helmholtz
    ax = axarr[5,1]
    ax.set_title(f"$\zeta$, Helmholtz GP")
    cs = ax.imshow(mean_vort_helm_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean_vort, vmax=max_mean_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #6-3: divergence standard
    ax = axarr[5,2]
    ax.set_title(f"$\zeta$, Velocity GP")
    cs = ax.imshow(mean_vort_std_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean_vort, vmax=max_mean_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #6-4: divergence dhnn
    ax = axarr[5,3]
    ax.set_title(f"$\zeta$, D-HNN")
    cs = ax.imshow(vort_dhnn, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean_vort, vmax=max_mean_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    ##############################

    #7-1: empty
    ax = axarr[6,0]
    ax.set_ylabel('Latitude');

    #7-2: divergence std helmholtz
    ax = axarr[6,1]
    ax.set_title(f"$\zeta$ (std), Helmholtz GP")
    cs = ax.imshow(std_vort_helm_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_std_vort, vmax=max_std_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #7-3: divergence std standard
    ax = axarr[6,2]
    ax.set_title(f"$\zeta$ (std), Velocity GP")
    cs = ax.imshow(std_vort_std_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_std_vort, vmax=max_std_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #7-4: empty
    ax = axarr[6,3]
   
   ##############################

    #8-1: empty
    ax = axarr[7,0]
    ax.set_ylabel('Latitude');
    ax.set_xlabel('Longitude');

    #8-2: z-score helmholtz
    ax = axarr[7,1]
    ax.set_title("$\zeta$ ($z$-value), Helmholtz GP")
    cs = ax.imshow(z_scores_helm_vort, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_z_vort, vmax=max_z_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Longitude'); 
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #8-3: z-score std
    ax = axarr[7,2]
    ax.set_title("$\zeta$ ($z$-value), Helmholtz GP")
    cs = ax.imshow(z_scores_std_vort, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_z_vort, vmax=max_z_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #8-4: empty
    ax = axarr[7,3]
    ax.set_xlabel('Longitude')

    remove_ticks(np.reshape(axarr, -1))

    rasterize_axes = lambda axarr: [a.set_rasterized(True) for a in axarr] 
    rasterize_axes(np.reshape(axarr, -1))

    plt.tight_layout()

    handles, labels = axarr[0, 1].get_legend_handles_labels()
    legend = f.legend(
        labels=labels,
        handles=handles,
        bbox_to_anchor=[0.5, 0.0005],
        loc="center",
        ncol=2,
        frameon=False,
        markerscale=3.0,
    )

    plt.savefig(f"{save_dest}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()

    return


######################

def plot_appendix_results_realdata(X_grid, Y_grid, XY_train, UV_train,
                                    test_mu_helm, test_cov_helm, 
                                    test_mu_std, test_cov_std,
                                    mean_div_helm, var_div_helm, 
                                    mean_div_std, var_div_std, 
                                    mean_vort_helm, var_vort_helm, 
                                    mean_vort_std, var_vort_std,
                                    pred_dhnn, 
                                    div_dhnn, vort_dhnn, 
                                    cmap='cool', cmap_vort='plasma', scale=3, scale_dots=0.1, save_dest = ""):
    """
    plot_appendix_results_realdata produces the plot that we include in the appendix of the paper to compare
    velocity, divergence and vorticity predictions, for Helmholtz GP, velocity GP, and D-HNN, for real data 
    (where we do not have the ground truth)

    Args:
        X_grid: np.array of shape [sqrt(N), sqrt(N)] with X coordinates of test observations on a grid
        Y_grid: np.array of shape [sqrt(N), sqrt(N)] with Y coordinates of test observations on a grid
        XY_train: np.array of shape [M, 2] with X & Y coordinates of observations
        UV_train: np.array of shape [M, 2] with U & V flow observations
        test_mu_helm: np.array of shape [N, 2] with Helmholtz GP posterior mean at test points
        test_cov_helm: np.array of shape [2N, 2N] with Helmholtz GP posterior covariance at test points
        test_mu_std: np.array of shape [N, 2] with Velocity GP posterior mean at test points
        test_cov_std: np.array of shape [2N, 2N] with Velocity GP posterior covariance at test points
        mean_div_helm: np.array of shape [N, 1] with Helmholtz GP posterior mean for divergence at test points
        var_div_helm: np.array of shape [N, 1] with Helmholtz GP posterior variance for divergence at test points
        mean_div_std: np.array of shape [N, 1] with Velocity GP posterior mean for divergence at test points
        var_div_std: np.array of shape [N, 1] with Velocity GP posterior variance for divergence at test points
        mean_vort_helm: np.array of shape [N, 1] with Helmholtz GP posterior mean for vorticity at test points
        var_vort_helm: np.array of shape [N, 1] with Helmholtz GP posterior variance for vorticity at test points
        mean_vort_std: np.array of shape [N, 1] with Velocity GP posterior mean for vorticity at test points
        var_vort_std: np.array of shape [N, 1] with Velocity GP posterior variance for vorticity at test points
        pred_dhnn: np.array of shape [N, 2] with D-HNN predicted velocity field
        div_dhnn: np.array of shape [N, 1] with D-HNN predicted divergence
        vort_dhnn: np.array of shape [N, 1] with D-HNN predicted vorticity
        cmap: color map for divergence plots
        cmap_vort: color map for vorticity plots
        scale: scale for plotting the arrows
        scale_dots: scale for plotting the dots for the real data buoys
        save_dest: path where you want your file to be saved in .pdf format
    """

    grid_points_hor = X_grid.shape[0]
    grid_points_ver = X_grid.shape[1]

    # format prediction mean and variance for plotting
    # helmholtz
    test_var_helm = torch.linalg.diagonal(test_cov_helm)
    test_mu_grid_helm = torch.reshape(test_mu_helm, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_var_helm = torch.reshape(test_var_helm, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_std_grid_helm = np.sqrt(test_var_helm[0]) + np.sqrt(test_var_helm[1])
    # standard
    test_var_std = torch.linalg.diagonal(test_cov_std)
    test_mu_grid_std = torch.reshape(test_mu_std, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_var_std = torch.reshape(test_var_std, [2, grid_points_hor, grid_points_ver]).detach().numpy()
    test_std_grid_std = np.sqrt(test_var_std[0]) + np.sqrt(test_var_std[1])

    #minor preprocessing to get the data in the desired form
    pred_dhnn = pred_dhnn.detach().numpy()
    vort_dhnn = torch.reshape(vort_dhnn, [grid_points_hor, grid_points_ver]).detach().numpy()
    div_dhnn = torch.reshape(div_dhnn, [grid_points_hor, grid_points_ver]).detach().numpy()
    
    #reshape d-hnn predictions
    u_dhnn = pred_dhnn[:,0].reshape(X_grid.shape)
    v_dhnn =pred_dhnn[:,1].reshape(X_grid.shape)

    # format mean and variance divergence for plotting
    # helmholtz
    mean_div_helm_grid = torch.reshape(mean_div_helm, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_div_helm_grid = np.sqrt(torch.reshape(var_div_helm, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_helm = (mean_div_helm_grid)/std_div_helm_grid
    # standard
    mean_div_std_grid = torch.reshape(mean_div_std, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_div_std_grid = np.sqrt(torch.reshape(var_div_std, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_std = (mean_div_std_grid)/std_div_std_grid

    # get limit values for divergence plotting on same scales
    # mean
    min_mean_helm, max_mean_helm = np.min(mean_div_helm_grid), np.max(mean_div_helm_grid)
    min_mean_std, max_mean_std = np.min(mean_div_std_grid), np.max(mean_div_std_grid)
    min_mean, max_mean = np.min([min_mean_helm, min_mean_std]), np.max([max_mean_helm, max_mean_std])
    # std
    min_std_helm, max_std_helm = np.min(std_div_helm_grid), np.max(std_div_helm_grid)
    min_std_std, max_std_std = np.min(std_div_std_grid), np.max(std_div_std_grid)
    min_std, max_std = np.min([min_std_helm, min_std_std]), np.max([max_std_helm, max_std_std])
    # z-scores
    min_z_helm, max_z_helm = np.min(z_scores_helm), np.max(z_scores_helm)
    min_z_std, max_z_std = np.min(z_scores_std), np.max(z_scores_std)
    min_z, max_z = np.min([min_z_helm, min_z_std]), np.max([max_z_helm, max_z_std])

    # format mean and variance vorticity for plotting
    # helmholtz
    mean_vort_helm_grid = torch.reshape(mean_vort_helm, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_vort_helm_grid = np.sqrt(torch.reshape(var_vort_helm, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_helm_vort = (mean_vort_helm_grid)/std_vort_helm_grid
    # standard
    mean_vort_std_grid = torch.reshape(mean_vort_std, [grid_points_hor, grid_points_ver]).detach().numpy()
    std_vort_std_grid = np.sqrt(torch.reshape(var_vort_std, [grid_points_hor, grid_points_ver]).detach().numpy())
    z_scores_std_vort = (mean_vort_std_grid)/std_vort_std_grid

    # get limit values for divergence plotting on same scales
    # mean
    min_mean_helm_vort, max_mean_helm_vort = np.min(mean_vort_helm_grid), np.max(mean_vort_helm_grid)
    min_mean_std_vort, max_mean_std_vort = np.min(mean_vort_std_grid), np.max(mean_vort_std_grid)
    min_mean_vort, max_mean_vort = np.min([min_mean_helm_vort, min_mean_std_vort]), np.max([max_mean_helm_vort, max_mean_std_vort])
    # std
    min_std_helm_vort, max_std_helm_vort = np.min(std_vort_helm_grid), np.max(std_vort_helm_grid)
    min_std_std_vort, max_std_std_vort = np.min(std_vort_std_grid), np.max(std_vort_std_grid)
    min_std_vort, max_std_vort = np.min([min_std_helm_vort, min_std_std_vort]), np.max([max_std_helm_vort, max_std_std_vort])
    # z-scores
    min_z_helm_vort, max_z_helm_vort = np.min(z_scores_helm_vort), np.max(z_scores_helm_vort)
    min_z_std_vort, max_z_std_vort = np.min(z_scores_std_vort), np.max(z_scores_std_vort)
    min_z_vort, max_z_vort = np.min([min_z_helm_vort, min_z_std_vort]), np.max([max_z_helm_vort, max_z_std_vort])

    ##############################

    # Plot results
    f, axarr = plt.subplots(nrows=7, ncols=3, figsize=[textwidth,1.28*textwidth], sharex='row', sharey='col')

    #1-1: prediction helmholtz
    ax = axarr[0,0]
    ax.set_title("$F$, Helmholtz GP")
    ax.quiver(X_grid, Y_grid, test_mu_grid_helm[0], test_mu_grid_helm[1], scale_units='xy', scale=scale, label="predicted currents")
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="buoys' locations")
    ax.set_ylabel('Latitude')
    #ax.legend(loc='upper right')

    #1-2: prediction standard
    ax = axarr[0,1]
    ax.set_title("$F$, Velocity GP")
    ax.quiver(X_grid, Y_grid, test_mu_grid_std[0], test_mu_grid_std[1], scale_units='xy', scale=scale, label="predicted currents")
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="buoys' locations")
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc='upper right')

    #1-3: prediction dhnn
    ax = axarr[0,2]
    ax.set_title("$F$, D-HNN")
    ax.quiver(X_grid, Y_grid, u_dhnn, v_dhnn, scale_units='xy', scale=scale, label="predicted currents")
    ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=scale, label="buoys' locations")
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc='upper right')

    
    ##############################

    #3-1: divergence helmholtz
    ax = axarr[1,0]
    ax.set_title(f"$\delta$, Helmholtz GP")
    cs = ax.imshow(mean_div_helm_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    ax.set_ylabel('Latitude');
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #3-2: divergence standard
    ax = axarr[1,1]
    ax.set_title(f"$\delta$, Velocity GP")
    cs = ax.imshow(mean_div_std_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #3-3: divergence dhnn
    ax = axarr[1,2]
    ax.set_title(f"$\delta$, D-HNN")
    cs = ax.imshow(div_dhnn, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean, vmax=max_mean, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    ##############################

    #4-1: divergence std helmholtz
    ax = axarr[2,0]
    ax.set_title(f"$\delta$ (std), Helmholtz GP")
    cs = ax.imshow(std_div_helm_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_std, vmax=max_std, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    ax.set_ylabel('Latitude');
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #4-2: divergence std standard
    ax = axarr[2,1]
    ax.set_title(f"$\delta$ (std), Velocity GP")
    cs = ax.imshow(std_div_std_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_std, vmax=max_std, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #4-3: empty
    ax = axarr[2,2]
   
   ##############################

    #5-1: z-score helmholtz
    ax = axarr[3,0]
    ax.set_title("$\delta$ ($z$-values), Helmholtz GP")
    cs = ax.imshow(z_scores_helm, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_z, vmax=max_z, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    ax.set_ylabel('Latitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #5-2: z-score std
    ax = axarr[3,1]
    ax.set_title("$\delta$ ($z$-values), Velocity GP")
    cs = ax.imshow(z_scores_std, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_z, vmax=max_z, cmap=cmap, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #5-3: empty
    ax = axarr[3,2]

    ##############################

    #6-1: vorticity helmholtz
    ax = axarr[4,0]
    ax.set_title(f"$\zeta$, Helmholtz GP")
    cs = ax.imshow(mean_vort_helm_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean_vort, vmax=max_mean_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    ax.set_ylabel('Latitude')#; ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #6-2: vorticity standard
    ax = axarr[4,1]
    ax.set_title(f"$\zeta$, Velocity GP")
    cs = ax.imshow(mean_vort_std_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean_vort, vmax=max_mean_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #6-3: vorticity dhnn
    ax = axarr[4,2]
    ax.set_title(f"$\zeta$, D-HNN")
    cs = ax.imshow(vort_dhnn, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_mean_vort, vmax=max_mean_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    ##############################

    #7-1: divergence std helmholtz
    ax = axarr[5,0]
    ax.set_title(f"$\zeta$ (std), Helmholtz GP")
    cs = ax.imshow(std_vort_helm_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_std_vort, vmax=max_std_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    ax.set_ylabel('Latitude');# ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #7-2: divergence std standard
    ax = axarr[5,1]
    ax.set_title(f"$\zeta$ (std), Velocity GP")
    cs = ax.imshow(std_vort_std_grid, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_std_vort, vmax=max_std_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    #ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #7-3: empty
    ax = axarr[5,2]
   
   ##############################

    #8-1: z-score helmholtz
    ax = axarr[6,0]
    ax.set_title("$\zeta$ ($z$-values), Helmholtz GP")
    cs = ax.imshow(z_scores_helm_vort, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_z_vort, vmax=max_z_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    ax.set_ylabel('Latitude'); ax.set_xlabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #8-2: z-score std
    ax = axarr[6,1]
    ax.set_title("$\zeta$ ($z$-values), Velocity GP")
    cs = ax.imshow(z_scores_std_vort, origin="lower", extent=(np.min(X_grid), np.max(X_grid), np.min(Y_grid), np.max(Y_grid)), vmin=min_z_vort, vmax=max_z_vort, cmap=cmap_vort, aspect="auto")
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=scale_dots)
    ax.set_xlabel('Longitude')
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cs, cax=cax)

    #8-3: empty
    ax = axarr[6,2]
    ax.set_xlabel('Longitude')

    remove_ticks(np.reshape(axarr, -1))
    rasterize_axes = lambda axarr: [a.set_rasterized(True) for a in axarr] 
    rasterize_axes(np.reshape(axarr, -1))
    plt.tight_layout()

    handles, labels = axarr[0, 1].get_legend_handles_labels()
    legend = f.legend(
        labels=labels,
        handles=handles,
        bbox_to_anchor=[0.5, 0.0005],
        loc="center",
        ncol=2,
        frameon=False,
        markerscale=3.0,
    )

    plt.savefig(f"{save_dest}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
    return
    
###############################

# Here we have utils function used in the rest of the .py file

def _to_numpy(lst):
    return [t.detach().numpy() if isinstance(t, torch.Tensor) else t for t in lst]

def remove_ticks(axes):
    for a in axes:
        a.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
        )

def add_colorbar(ax, cs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad="2%")
    cbar = plt.colorbar(cs, cax=cax)
    cbar.ax.tick_params(labelsize=7.0)
    cbar.ax.locator_params(nbins=4)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


#############################################
