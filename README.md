Accompagnying repository for 'Quantum machine learning-based parameterization for atmospheric boundary layer turbulence'

### Content

Notebooks and scripts:
- coarse graining including calculation of the subgridscale flux correction: coarsegraining.py
- Figures 1 and 2: LES_data_figures.ipynb, Smagorinsky.ipynb
- data preprocessing: preprocessing.ipynb
- QNN model training: training.ipynb
- cl. NN training: training_classical_nn.ipynb
- Figure 3: performance_eval.ipynb
- Figure 4: generalisation.ipynb
- Figure 5: shapley_values.ipynb
- Appendix: qnn_hyperparams.ipynb

Folders:
- libs: circuits.py (QNN circuit layouts), qnn_models.py (QNN models based on circuits in circuits.py and training), helpers.py (helper for plotting), dataloading.py (functions to load and preprocess the training data) 
- abl_heights: data for boundary layer top as part of preprocessing
- spectra: spectra from Figure 2
- figures: pdf files of the figures
- QNN_results: performance and output data of the QNN
- cl_results: performance and output data of the cl. NN

Other files:
- Smagorinsky.txt: prediction from applying Smagorinsky closure to the coarse data
- coarsegraining.py: script for coarse graining the high-resolution data

### Data
The coarse-grained Large Eddy Simulation data on which this work is based, and details about the high-resolution data can be found at 10.5281/zenodo.20326042.

### Environment

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