# Gait in Aging and Disease Dataset (GADD)
This repository includes the code for Gait in Aging and Disease Dataset (GADD).

## Raw Data
For the raw data of abnormal participant, see `ptbdb_abnormal.csv`.
For the raw data of normal participant, see `ptbdb_normal.csv`.
For the visualisation of the raw data, see `plot_original_data_normal_1.png`, `plot_original_data_normal_2.png`, `plot_original_data_abnormal_1.png`, and `plot_original_data_abnormal_2.png`, the code is in `generate_entropy.ipynb`.
For the selection of the entropy features, see `EntropySelection1.png` and `EntropySelection2.png`, the code is in `generate_entropy.ipynb`.

## Main Step
1. Generate baseline features and entropy features, shown as `generate_entropy.ipynb`.
2. Build Linear Regression model by baseline features and entropy features, shown as `model_lr_baseline.ipynb` and `model_lr_entropy.ipynb`, the results are shown as `evaluation_baseline_LR.csv` and `evaluation_entropy_LR.csv`.
3. Build SVM model by baseline features and entropy features, shown as `model_svm_baseline.ipynb` and `model_svm_entropy.ipynb`, the results are shown as `evaluation_baseline_SVM.csv` and `evaluation_entropy_SVM.csv`.
4. Build MLP model by baseline features and entropy features, shown as `network_pytorch`, the results are shown as `evaluation_baseline_MLP.csv` and `evaluation_entropy_MLP.csv`.