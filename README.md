
## Note: The code in this repo is an older version, the new version will be updated soon...


# Analyzing Entropy Features in Time-Series Data for Pattern Recognition in Neurological Conditions
This repository includes the code required to reproduce the experiments and figures in the pre-print paper "Analysing Entropy Features in Time-Series Data for Pattern Recognition in Neurological Conditions".

## 1. Requirements
This directory contains the code required to run the experiments to produce the results presented in the paper "Information Theory Inspired Pattern Analysis for Time-series Data"

To get started and download all dependencies, run:

```
pip install -r requirements.txt 
```
## 2. Minder dataset
### 2.1 Raw Data
The Minder dataset is privacy.
To apply and download the dataset, please contact with [Prof. Payam Barnaghi](mailto:p.barnaghi@imperial.ac.uk).

### 2.2 Visualisation of the pre-processed data

<img src="./MINDER/Figures/raw_data.png" width="800"> 


### 2.3 Feature generation and selection

Generate entropy features, shown as `EntropyFeatures`.

Entropy features for the Minder dataset includes: 
Entropy of Markov chains: `./MINDER/EntropyFeatures/activity_daytime_per_week_mk_entropy.ipynb`, `./MINDER/EntropyFeatures/activity_night_per_week_mk_entropy.ipynb`  


Entropy rate of Markov chains: `./MINDER/EntropyFeatures/activity_daytime_per_week_mk_entropy_rate.ipynb`, `./MINDER/EntropyFeatures/activity_night_per_week_mk_entropy_rate.ipynb`  


Entropy production of Markov chains: `./MINDER/EntropyFeatures/activity_daytime_per_week_mk_entropy_production.ipynb`, `./MINDER/EntropyFeatures/activity_night_per_week_mk_entropy_production.ipynb`  


Von Neumann Entropy of Markov chains (activity frequency): `./MINDER/EntropyFeatures/activity_daytime_per_week_mk_vn_entropy_frequency.ipynb`, `./MINDER/EntropyFeatures/activity_night_per_week_mk_vn_entropy_frequency.ipynb`  


Von Neumann Entropy of Markov chains (activity duration): `./MINDER/EntropyFeatures/activity_daytime_per_week_mk_vn_entropy_duration.ipynb`, `./MINDER/EntropyFeatures/activity_night_per_week_mk_vn_entropy_duration.ipynb` 


The baseline features: `./MINDER/EntropyFeatures/activity_daytime_night_per_week_frequency.ipynb`

### 2.4 Modelling and results

The evaluation results are:

<img src="./MINDER/Figures/results.png" width="1000" title="The evaluation results">




## 3. Epileptic Seizure Recognition Dataset (ESRD)
This repository includes the code for Epileptic Seizure Recognition Dataset (ESRD).

### 3.1 Raw Data
Download the [orginal data](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)[1]. 

### 3.2 Visualisation of the pre-processed data

<img src="./ESRD/Figures/plot_result_raw.png" width="400"> 

The code is in `./ESRD/plot_raw.ipynb`.

### 3.3 Feature generation and selection
Generate entropy features, shown as `./ESRD/generate_entropy.ipynb`.

Select entropy features by Pearson relationship matrix and mutual information.

<img src="./ESRD/Figures/EntropySelection1.png" width="400" alt="Pearson relationship matrix">  <img src="./ESRD/Figures/EntropySelection2.png" width="400" alt="Mutual information">

### 3.4 Modelling and results
#### (1) Baseline-CNN
The code is shown in `./ESRD/model_baseline_CNN.ipynb`.

#### (2) Baseline-LSTM
The code is shown in `./ESRD/model_baseline_LSTM.ipynb`.

#### (3) Entropy-MLP
The code is shown in `./ESRD/network_pytorch`.

The evaluation results are:

<img src="./ESRD/Figures/plot_result_all.png" width="300" title="The evaluation results">


[1]Andrzejak, R.G., Lehnertz, K., Mormann, F., Rieke, C., David, P. and Elger, C.E., 2001. Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. Physical Review E, 64(6), p.061907.

## 4. PTB Diagnostic ECG Database (PTBDB)
This repository includes the code for PTB Diagnostic ECG Database.

### 4.1 Raw Data
Download the [orginal data](https://www.physionet.org/content/ptbdb/1.0.0/)[2]. 

Download the [pre-processed data](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

Here we utilize the pre-processed data.

### 4.2 Visualisation of the pre-processed data
For the normal participants:

<img src="./PTBDB/Figures/plot_original_data_normal_all.png" width="300"> 

For the abnormal participants:

<img src="./PTBDB/Figures/plot_original_data_abnormal_all.png" width="300">

The code is in `./PTBDB/generate_entropy.ipynb`.

### 4.3 Feature generation and selection
Generate entropy features, shown as `./PTBDB/generate_entropy.ipynb`.

Select entropy features by Pearson relationship matrix and mutual information.

<img src="./PTBDB/Figures/EntropySelection1.png" width="400" alt="Pearson relationship matrix">  <img src="./PTBDB/Figures/EntropySelection2.png" width="400" alt="Mutual information">

### 4.4 Modelling and results
####  (1) Baseline-CNN
The code is shown in `./PTBDB/model_baseline_CNN.ipynb`.

####  (2) Baseline-MLP
The code is shown in `./PTBDB/model_baseline_MLP.ipynb`.

####  (3) Entropy-MLP
The code is shown in `./PTBDB/network_pytorch`.

The evaluation results are:

<img src="./PTBDB/Figures/plot_result_all.png" width="300" title="The evaluation results">

The code is shown in `./PTBDB/plot_result.ipynb`.

[1]Andrzejak, R.G., Lehnertz, K., Mormann, F., Rieke, C., David, P. and Elger, C.E., 2001. Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. Physical Review E, 64(6), p.061907.


[2]Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215 - e220.
