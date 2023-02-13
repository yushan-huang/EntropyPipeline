# Epileptic Seizure Recognition Dataset (ESRD)
This repository includes the code for Epileptic Seizure Recognition Dataset (ESRD).

## Raw Data
Download the [orginal data](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)[1]. 

## Visualisation of the pre-processed data

<img src="./Figures/plot_result_raw.png" width="300"> 

The code is in `plot_raw.ipynb`.

## Feature generation and selection
Generate entropy features, shown as `generate_entropy.ipynb`.

Select entropy features by Pearson relationship matrix and mutual information.

<img src="./Figures/EntropySelection1.png" width="400" alt="Pearson relationship matrix">  <img src="./Figures/EntropySelection2.png" width="400" alt="Mutual information">

## Modelling and results
### Baseline-CNN
The code is shown in `model_baseline_CNN.ipynb`.

### Baseline-LSTM
The code is shown in `model_baseline_LSTM.ipynb`.

### Entropy-MLP
The code is shown in `network_pytorch`.

The evaluation results are:

<img src="./Figures/plot_result_all.png" width="300" title="The evaluation results">


[1]Andrzejak, R.G., Lehnertz, K., Mormann, F., Rieke, C., David, P. and Elger, C.E., 2001. Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. Physical Review E, 64(6), p.061907.