# Vacasa

## Overview
This project implements a machine learning pipeline in Python, designed to process hotel booking data, 
train models, evaluate their performance, and use them for predictions. The pipeline comprises several 
components, including data preprocessing, model training and evaluation, model promotion, and prediction. 
Additionally, Jupyter notebooks are provided for a data science exploration.

## Requirements:
* Python 3.12
* libraries - see requirements.txt
* Jupyter Notebooks (for the data science part)
## Project Structure

* notebooks: Jupyter notebook for data exploration and analysis.
* argument_parser.py: Handles command-line arguments for model selection and parameter tuning.
* preprocessing.py: Contains preprocessing steps for the input data.
* promotion.py: Manages the logic for model promotion based on performance metrics.
* train.py: Script for training models, evaluating their performance, and promoting them based on improvement criteria.
* utils.py: Provides utility functions for data loading and other common tasks.
* predict.py: Script for making predictions using the promoted model.
 
## Getting Started

### Installation
Ensure that Python 3.12, Jupyter, and all required libraries are installed. You can install the libraries using:
```
pip install -r requirements.txt
```
### Data Science Exploration
To explore the data science aspects, navigate to the notebooks directory and open the Jupyter notebooks.

### Training a Model
To train a model, run the train.py script with the necessary arguments. You can look all available commands by:
```
python train.py -h
```
### Making Predictions
To make predictions, use the predict.py script.
```
python predict.py --infile <PATH_TO_TEST_DATA>
```