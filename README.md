# Information Theory Inspired Pattern Analysis for Time-series Data
This repository includes the code required to reproduce the experiments and figures in the pre-print paper "Information Theory Inspired Pattern Analysis for Time-series Data".
## Requirements
This directory contains the code required to run the experiments to produce the results presented in the paper "Information Theory Inspired Pattern Analysis for Time-series Data"

To get started and download all dependencies, run:

```
pip install -r requirements.txt 
```
## Minder dataset
See `Minder`.

## Gait in Aging and Disease Dataset (GADD)
See `GADD`.

## PTB Diagnostic ECG Database (PTBDB)

Details are in `PTBDB`.

### Raw Data
Download the [orginal data](https://www.physionet.org/content/ptbdb/1.0.0/)[1]. 

Download the [pre-processed data](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

### Visualisation of the pre-processed data
For the normal participants:

<img src="./PTBDB/Figures/plot_original_data_normal_1.png" width="300"> <img src="./PTBDB/Figures/plot_original_data_normal_2.png" width="300">

For the abnormal participants:

<img src="./PTBDB/Figures/plot_original_data_abnormal_1.png" width="300"> <img src="./PTBDB/Figures/plot_original_data_abnormal_2.png" width="300">

The code is in `generate_entropy.ipynb`.

### Feature generation and selection
Generate baseline features and entropy features, shown as `generate_entropy.ipynb`.

Select entropy features by Pearson relationship matrix and mutual information.

<img src="./PTBDB/Figures/EntropySelection1.png" width="400" alt="Pearson relationship matrix">  <img src="./PTBDB/Figures/EntropySelection2.png" width="400" alt="Mutual information">

### Modelling and results
#### Logistic regression (LR)
The code is shown in `model_lr_baseline.ipynb` and `model_lr_entropy.ipynb`.

The evaluation results are:

<img src="./PTBDB/Figures/plot_result_LR.png" width="300" title="The evaluation results of LR">

#### Support vector machine (SVM)
The code is shown in `model_svm_baseline.ipynb` and `model_svm_entropy.ipynb`.

The evaluation results are:

<img src="./PTBDB/Figures/plot_result_SVM.png" width="300" title="The evaluation results of SVM">

#### Multilayer perceptron (MLP)
The code is shown in `network_pytorch`.

The evaluation results are:

<img src="./PTBDB/Figures/plot_result_SVM.png" width="300" title="The evaluation results of MLP">


[1]Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215 - e220.
