# Exoplanet Hunting in Deep Space

This repo explore and model the dataset of data provided by Kaggle with the intent to find stars with at least one exoplanet orbiting around. The data describe the change in flux (light intensity) of several thousand stars. Each star has a binary label of 2 or 1. 2 indicated that that the star is confirmed to have at least one exoplanet in orbit; some observations are in fact multi-planet systems.

<br/>
<p align="center">
  <img src="/images/Kepler-720x340.jpeg" width=50% height=50% />
</p>
<br/>

Planets themselves do not emit light, but the stars that they orbit do. If said star is watched over several months or years, there may be a regular 'dimming' of the flux (the light intensity). This is evidence that there may be an orbiting body around the star.

## Project Overview

The repository contains two main notebooks:

__data_explotation.ipynb__ This first notebook aims to explore the data and get some important insights about the dataset. Specifically a visualization of how the time-series looks like has shown how (often) stars with exoplanets and stars without exoplanets have different range of flux fluctuation over time. This propriety can be levered by a ML/DL model to distinguish the two classes.
<p align="center">
  <img src="/images/exoplanets.png" width=100% height=100% />
  <img src="/images/Non-exoplanets.png" width=100% height=100% />
</p>


__model_train.ipynb__ The second notebook contains all the code used to model the data with SVC - Support Vector Classifier used as baseline model and a CNN coded with TensorFlow. The latter are reported in the final notes (at the botton of the notebook) performs better since can better explore the features provided.
<img src="/images/loss_function.png" width=100% height=100% />

## Project Instructions

In order to run the model some basic libraries are needed. Run the following command-lines to create a new conda environment and install the required libraries.

```
conda create -n exo_hunting python=3.6 -y
conda activate exo_hunting
pip install numpy, tensorflow-gpu, seaborn, scikit-learn
```

## Project Data

The data used in the repo is provided by Kaggle: [Exoplanet Hunting in Deep Space](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data).

It can be downloaded with Kaggle API with the following command:
```
kaggle datasets download -d keplersmachines/kepler-labelled-time-series-data
```

## Possible Improvements

- Evaluate XGBoost algorithm and DNN algorithm over the dataset.
- Create model package to train TensorFlow model over GCP ML engine.
- Perform HPO on the model
