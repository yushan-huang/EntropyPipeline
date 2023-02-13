# Gait in Aging and Disease Dataset (GADD)
This repository includes the code for Gait in Aging and Disease Dataset (GADD).

## Raw Data
Download the [orginal data](https://www.physionet.org/content/ptbdb/1.0.0/)[1]. 

Download the [pre-processed data](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

## Visualisation of the pre-processed data
For the normal participants:

<img src="./Figures/plot_original_data_normal_all.png" width="300"> 

For the abnormal participants:

<img src="./Figures/plot_original_data_abnormal_all.png" width="300">

The code is in `generate_entropy.ipynb`.

## Feature generation and selection
Generate entropy features, shown as `generate_entropy.ipynb`.

Select entropy features by Pearson relationship matrix and mutual information.

<img src="./Figures/EntropySelection1.png" width="400" alt="Pearson relationship matrix">  <img src="./Figures/EntropySelection2.png" width="400" alt="Mutual information">

## Modelling and results
### Baseline-CNN
The code is shown in `model_baseline_CNN.ipynb`.

### Baseline-MLP
The code is shown in `model_baseline_MLP.ipynb`.

### Entropy-MLP
The code is shown in `network_pytorch`.

The evaluation results are:

<img src="./Figures/plot_result_all.png" width="300" title="The evaluation results">


[1]Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215 - e220.