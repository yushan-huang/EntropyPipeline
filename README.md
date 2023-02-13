# Information Theory Inspired Pattern Analysis for Time-series Data
This repository includes the code required to reproduce the experiments and figures in the pre-print paper "Information Theory Inspired Pattern Analysis for Time-series Data".
## 1. Requirements
This directory contains the code required to run the experiments to produce the results presented in the paper "Information Theory Inspired Pattern Analysis for Time-series Data"

To get started and download all dependencies, run:

```
pip install -r requirements.txt 
```
## 2. Minder dataset
See `Minder`.

## 3. Gait in Aging and Disease Dataset (GADD)
See `GADD`.

## 4. PTB Diagnostic ECG Database (PTBDB)
This repository includes the code for PTB Diagnostic ECG Database.

### 4.1 Raw Data
Download the [orginal data](https://www.physionet.org/content/ptbdb/1.0.0/)[1]. 

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

<img src="./Figures/EntropySelection1.png" width="400" alt="Pearson relationship matrix">  <img src="./Figures/EntropySelection2.png" width="400" alt="Mutual information">

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


[1]Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215 - e220.
