# Quantum machine learning-based parameterization for atmospheric boundary layer turbulence

This repository provides the code for developing, training, and evaluating data-driven hybrid quantum and classical atmospheric turbulence parameterizations based on large eddy simulation (LES) experiments. The models are compared with regard to performance, generalisation and interpretability properties. The hybrid quantum models, which are based on parameterized circuits implemented with Pennylane (Ville Bergholm et al. PennyLane: Automatic differentiation of hybrid quantum-classical computations. 2018. [arXiv:1811.04968](https://arxiv.org/abs/1811.04968)), show similar performance and generalisation properties to classical models of comparable size. The feature importances quantified through SHAPLEY values (Scott M Lundberg and Su-In Lee, A Unified Approach to Interpreting Model Predictions, [Advances in Neural Information Processing Systems 30 (NIPS 2017)](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)) are more stable in the quantum models with respect to initialization of the weights and variational parameters, outlining possible advantages in physical stability and generalisability.

The current release on zenodo can be found here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1273286464.svg)](https://zenodo.org/badge/latestdoi/1273286464)

### Data
The coarse-grained LES data on which this work is based, and details about the high-resolution data can be found in the data repository 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20326042.svg)](https://zenodo.org/badge/latestdoi/20326042)

The data were generated with the PALM model:

Maronga, B., et al (2020): Overview of the PALM model system 6.0, [Geosci. Model Dev., 13, 1335–1372](https://doi.org/10.5194/gmd-13-1335-2020)

Maronga, B., Gryschka, M., Heinze, R., Hoffmann, F., Kanani-Sühring, F., Keck, M., Ketelsen, K., Letzel, M. O., Sühring, M., and Raasch, S. (2015): The Parallelized Large-Eddy Simulation Model (PALM) version 4.0 for atmospheric and oceanic flows: model formulation, recent developments, and future perspectives, [Geosci. Model Dev., 8, 1539-1637](https://doi.org/10.5194/gmd-8-2515-2015) 

## Content of the repository

### Notebooks and scripts:
- coarse graining including calculation of the subgridscale flux correction: coarsegraining.py
- Figures 1 and 2: LES_data_figures.ipynb, Smagorinsky.ipynb
- data preprocessing: preprocessing.ipynb
- QNN model training: training.ipynb
- cl. NN training: training_classical_nn.ipynb
- Figure 3: performance_eval.ipynb
- Figure 4: generalisation.ipynb
- Figure 5: shapley_values.ipynb
- Appendix: qnn_hyperparams.ipynb

### Folders:
- libs: circuits.py (QNN circuit layouts), qnn_models.py (QNN models based on circuits in circuits.py and training), helpers.py (helper for plotting), dataloading.py (functions to load and preprocess the training data) 
- abl_heights: data for boundary layer top as part of preprocessing
- spectra: spectra from Figure 2 (generate data in notebook)
- figures: pdf files of the figures (generate in notebooks)
- QNN_results: performance and output data of the QNN (generate data in notebooks)
- cl_results: performance and output data of the cl. NN (generate data in notebooks)

### Other files:
- Smagorinsky.txt: prediction from applying Smagorinsky closure to the coarse data (generate data in notebook)
- coarsegraining.py: script for coarse graining the high-resolution data



## Dependencies

The environement to run the codes and create a kernel for the notebooks can be created as:
```
conda env create --name my_env --file env.yml
conda activate my_env
```
In case the environment file ```env.yml``` is not working (most likely because some of the dependencies are not available anymore), we suggest creating the environment with the following key dependencies:
```
conda create -n my_env python=3.11 numpy matplotlib scipy pandas xarray sklearn pennylane jax keras
```

Install anaconda ipykernel (if needed) and create a new kernel for Jupyter Notebook

```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=my_env
```
