# PF-TKRR
## Intro

This is the public repository for the paper: Efficient Patient-Finetuned Seizure Detection with Tensor Kernel Ridge Regression.

## Required packages
### Standard packages
The required python packages can be found in [requirements.txt](requirements.txt).
It is recommended to first create a conda environment to install the packages and required python version. This can be done with the [environment.yml](environment.yml) file, which also installs the conda packages. Note that the remaining packages still need to be installed with pip.

### Additional packages

For pre-processing the `seizure_data_processing` package is needed: https://github.com/sderooij/seizure_data_processing/releases/tag/v0.0.1

In addition to that also the (self-created) `TensorLibrary` package is required, as it contains the CP-KRR model: https://github.com/sderooij/tensorlibrary/releases/tag/v0.0.1.


## Usage
Before running the scripts and notebooks please modify the `src/config.py` file.
The order in which the scripts and notebooks need to be run is as follows:
1. `find_tle_patients.ipynb`
2. `feature_exstraction.py`
3. `create_set.ipynb`
4. `preprocess.py`
5. `classification.py` (or run `classify.sh`)
6. `fine_tune.py`
7. `post_process.py`
8. `visualize_results.py`
9. `convergence.py`
10. `convergence_auc_plot.ipynb`

Explanation on their usage is (in principle) provided in the respective files. Note that there may be issues with import due to different (working) paths

## Questions
For any questions you can contact: s.j.s.derooij@tudelft.nl



