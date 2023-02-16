# Helmholtz-GP

PyTorch implementation of [Gaussian processes at the Helm(holtz): A more fluid model for ocean currents](**Arxiv link**) by Renato Berlinghieri, Brian L. Trippe, David R. Burt, Ryan Giordano,
Kaushik Srinivasan, Tamay Özgökmen, Junfei Xia, and Tamara Broderick. 

We present a method for modelling ocean currents given buoy data with Gaussian Processes (GPs) and an Helmholtz decomposition. In this repo we provide the code and the results for the following experiments:
- Synthetic experiments: 
    1. Single vortex
    2. Single vortex with straight current 
    3. Single point of divergence, over small area
    4. Single point of divergence, over medium sized area
    5. Single point of divergence, over big area
    6. Duffing oscillator with two small areas of divergence
    7. Duffing oscillator with two medium areas of divergence
    8. Duffing oscillator with two big areas of divergence
- Real data experiments:
    1. [LASER](http://carthe.org/laser/) experiment ; with sparse data and less sparse data.
    2. [GLAD](http://carthe.org/glad/) experiment ; with sparse data and less sparse data.

The structure of the repository is as follows:

- The folder "data" contains the data used in our experiments. For each experiment, syntethic or real data, we provide training and test buoy locations (XY_train & XY_test), and training and test velocity observations (UV_train & UV_test), the grid of longitudes and latitudes (X_grid and Y_grid) of test points. Moreover, for the synthetic experiments, we provide the underlying known divergence (div_grid) and vorticity (vort_grid).

- In the folder "helmholtz_gp" you can find the files (i) "helmholtz_regression_pytorch.py", that contains all the functions used for running our GP regression through the Helmholtz decomposition, written in Python 3, (ii) "optimization_loop.py" and "parameters.py" which are files called for the optimization of the hyperparameters in each experiment, and (iii) "plot_helper_arxiv.py" which is imported for producing the plots in the paper.

- The folder "Notebooks" contain 12 .ipynb notebooks (one for each experiment stated in the list above) that show how to use the functions from "helmholtz_gp" to produce ocean currents prediction for simulation and real world dataset. 

- The folder "plots_arxiv" containing the plots currently included in the arxiv manuscript. 

## Basic usage

1. Install the package. Navigate to the package root directory and run `python3 -m pip install .` Dependencies for the main package should be installed automatically. See the note below regarding installation of the Dissipative Hamiltonian Neural Networks code for recreating experiments in the notebooks. 
2. Upload your data with the desired format and shape (as per the "helmholtz_regression_pytorch.py") documentation. 
3. Fit the Helmholtz GP and compute predictions (step 3 in the demo notebooks)
4. Fit the velocity GP and compute predictions (step 4 in the demo notebooks)
5. Compute posterior divergence/vorticity (if interested) for each method
6. Visualize the results

***Note***: in our paper, we compare the predictions with the ones obtained usign [Dissipative Hamiltonian Neural Networks](https://github.com/greydanus/dissipative_hnns). To obtain these, we recommend cloning the linked repo and following what we do in step 8 of our .ipynb notebooks.    


 
