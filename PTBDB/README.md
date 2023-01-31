# Gait in Aging and Disease Dataset (GADD)
This repository includes the code for Gait in Aging and Disease Dataset (GADD).

## Raw Data
Download the [orginal data](https://www.physionet.org/content/ptbdb/1.0.0/)[1]. 
Download the [pre-processed data](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

## Visualisation of the raw data
For the normal participants:
![](./Figures/plot_original_data_abnormal_1.png =50x50)
<!-- ![image info](./Figures/plot_original_data_abnormal_1.png) -->
<!-- ![Normal_1](/Figures/plot_original_data_normal_1.png)
![image info](./pictures/image.png)
![Normal_2](/Figures/plot_original_data_normal_2.png)

For the abnoraml participants:
![Noraml_1](/Figures/plot_original_data_abnormal_1.png)
![Noraml_2](/Figures/plot_original_data_abnormal_2.png) -->

{image} ../Figures/plot_original_data_normal_1.png
:alt: fishy
:class: bg-primary mb-1
:width: 200px
:align: center

The code is in `generate_entropy.ipynb`.

## Main Step
1. Generate baseline features and entropy features, shown as `generate_entropy.ipynb`.
2. Build Linear Regression model by baseline features and entropy features, shown as `model_lr_baseline.ipynb` and `model_lr_entropy.ipynb`, the results are shown as `evaluation_baseline_LR.csv` and `evaluation_entropy_LR.csv`.
3. Build SVM model by baseline features and entropy features, shown as `model_svm_baseline.ipynb` and `model_svm_entropy.ipynb`, the results are shown as `evaluation_baseline_SVM.csv` and `evaluation_entropy_SVM.csv`.
4. Build MLP model by baseline features and entropy features, shown as `network_pytorch`, the results are shown as `evaluation_baseline_MLP.csv` and `evaluation_entropy_MLP.csv`.

[1]Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215 - e220.