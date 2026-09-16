# Break-Through-Tech Capstone: Census Income Prediction

## Overview
This repository contains a Jupyter Notebook detailing the end-to-end machine learning lifecycle to solve a predictive business problem. The project involves building and comparing traditional supervised learning models and a neural network to predict whether an individual's annual income exceeds $50,000.

## Business Problem
This problem matters to organizations focused on helping underserved communities. By accurately predicting income brackets, these organizations can identify individuals who may qualify for low-interest loans and financial education programs, thereby optimizing outreach and maximizing community benefit.

## Dataset
**Source:** [1994 U.S. Census Data](https://archive.ics.uci.edu/dataset/20/census+income)  
**Target/Label:** `income_binary` (Binary classification: `<=50K` vs `>50K`)  
**Key Features:** `age`, `workclass`, `fnlwgt`, `education`, `marital-status`, `sex_selfID`, `hours-per-week`, `capital-gain`, and `capital-loss`.

## Project Contents (Inside the `.ipynb` Notebook)
The primary analysis is contained within the Jupyter Notebook, which encompasses the following steps:

1. **Data Loading & Problem Definition:** Initialization of the dataset and formulation of the business objective.
2. **Exploratory Data Analysis (EDA):** 
   - Analyzed summary statistics and feature distributions (e.g., age demographics heavily crowded on the left).
   - Evaluated missing values in columns like `workclass`, `occupation`, and `native-country`.
   - Identified and addressed a significant class imbalance in the income label (more low-income than high-income individuals).
   - Explored the impact of outliers in `capital-gain` and `capital-loss`.
3. **Data Preparation:** Feature engineering, handling missing data, and converting target string labels to boolean values.
4. **Traditional Machine Learning Modeling:** Training, testing, and evaluating models including:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree Classifier
5. **Deep Learning Modeling:** Implementation of a Neural Network using TensorFlow/Keras.
6. **Evaluation & Comparison:** Comparing the performance of traditional ML models against the neural network using metrics like Accuracy and F1-Score.

## Requirements
To run the notebook, ensure you have the following packages installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `tensorflow` (Required for the Neural Network section)

## How to Run
1. Clone this repository.
2. Ensure the `censusData.csv` dataset is located in the appropriate `../../data/` directory relative to the notebook, or update the file path in the notebook.
3. Install the dependencies.
4. Open the Jupyter Notebook and execute the cells sequentially.
