# Epileptic Seizure Recognition Dataset (ESRD)
This repository includes the code for Minder Dataset (ESRD).

## Raw Data
The Minder dataset is privacy.
To apply and download the dataset, please contact with [Prof. Payam Barnaghi](mailto:p.barnaghi@imperial.ac.uk)  

## Visualisation of the pre-processed data

<img src="./Figures/raw_data.png" width="800"> 


## Feature generation and selection

Generate entropy features, shown as `EntropyFeatures`.

Entropy features for the Minder dataset includes: 
Entropy of Markov chains: `./EntropyFeatures/activity_daytime_per_week_mk_entropy.ipynb`, `./EntropyFeatures/activity_night_per_week_mk_entropy.ipynb`  


Entropy rate of Markov chains: `./EntropyFeatures/activity_daytime_per_week_mk_entropy_rate.ipynb`, `./EntropyFeatures/activity_night_per_week_mk_entropy_rate.ipynb`  


Entropy production of Markov chains: `./EntropyFeatures/activity_daytime_per_week_mk_entropy_production.ipynb`, `./EntropyFeatures/activity_night_per_week_mk_entropy_production.ipynb`  


Von Neumann Entropy of Markov chains (activity frequency): `./EntropyFeatures/activity_daytime_per_week_mk_vn_entropy_frequency.ipynb`, `./EntropyFeatures/activity_night_per_week_mk_vn_entropy_frequency.ipynb`  


Von Neumann Entropy of Markov chains (activity duration): `./EntropyFeatures/activity_daytime_per_week_mk_vn_entropy_duration.ipynb`, `./EntropyFeatures/activity_night_per_week_mk_vn_entropy_duration.ipynb` 


The baseline features: `./EntropyFeatures/activity_daytime_night_per_week_frequency.ipynb`

## Modelling and results

The evaluation results are:

<img src="./Figures/results.png" width="800" title="The evaluation results">
