# COMP90049 Assignment 2: Short-form Videos Trend Prediction

This repository contains a collaborative machine learning research project for **COMP90049 Introduction to Machine Learning** at the University of Melbourne.

The project investigates how machine learning methods can be used to predict the popularity trajectory of short-form social media videos using metadata collected from platforms such as YouTube Shorts and TikTok.

---

# Project Overview

Short-form video platforms have rapidly become one of the dominant forms of online media consumption. Millions of videos are uploaded daily, and only a small proportion become highly popular or sustain long-term engagement. Understanding the factors that contribute to video trend behaviour has important applications in:

- recommendation systems,
- digital marketing,
- creator analytics,
- audience engagement prediction,
- and platform content optimisation.

This project aims to explore whether machine learning models can effectively predict the trend category of a short-form video using metadata and engagement-related features.

Rather than focusing only on maximising prediction accuracy, this project investigates how different feature groups and modelling approaches influence prediction performance, interpretability, and generalisation.

Dataset used:

- [YouTube Shorts and TikTok Trends 2025](https://www.kaggle.com/datasets/tarekmasryo/youtube-shorts-and-tiktok-trends-2025)

The dataset contains large-scale metadata relating to short-form videos, including information about:
- platform source,
- region,
- engagement statistics,
- content category,
- traffic source,
- creator-related attributes,
- and temporal trend behaviour.

The dataset is suitable for this project because it contains a sufficiently large number of samples and diverse features, allowing meaningful experimentation with:
- feature engineering,
- feature selection,
- classification models,
- neural networks,
- and comparative evaluation techniques.

---

# Main Research Problem

> How effectively can short-form video trends (rising, stable, declining, seasonal) be predicted using metadata and content-related features?

---

# Research Questions

As required for a 4-person group project, this study investigates multiple distinct research questions that examine different aspects of the prediction problem.

## **RQ1 — Influential Factors**

What factors, such as platform, region, content category, traffic source, and engagement metrics, most strongly influence video trend classification?

This question focuses on identifying which features contribute most significantly to prediction outcomes and understanding the relationship between metadata attributes and trend behaviour.

---

## **RQ2 — Feature Group Effectiveness**

How does the use of different feature groups, including content-related, platform-related, temporal, and creator-related features, affect prediction performance?

This question investigates the impact of feature engineering and feature selection on model quality and explores whether combining multiple feature categories improves predictive capability.

---

## **RQ3 — Model Comparison**

How do traditional machine learning models compare with neural network models in predicting short-form video trend labels?

This question evaluates the strengths and limitations of different modelling approaches across:
- classical machine learning methods,
- ensemble-based approaches,
- and neural network architectures.

Models explored may include:
- Logistic Regression,
- Decision Trees,
- Random Forests,
- Support Vector Machines,
- Gradient Boosting methods,
- and Multi-Layer Perceptron (MLP) neural networks.

---

# Planned Methodology

The project will follow a standard machine learning workflow consisting of:

1. **Data Cleaning and Preprocessing**
   - handling missing values,
   - encoding categorical variables,
   - scaling numerical features,
   - removing duplicates and outliers.

2. **Feature Engineering and Selection**
   - constructing engagement ratios,
   - temporal trend indicators,
   - interaction-based features,
   - and evaluating feature importance.

3. **Exploratory Data Analysis**
   - analysing class distributions,
   - identifying skewness and imbalance,
   - visualising correlations and trends.

4. **Model Training and Hyperparameter Tuning**
   - training multiple supervised learning models,
   - performing hyperparameter optimisation,
   - comparing bias-variance characteristics.

5. **Evaluation and Validation**
   - train/validation/test splitting,
   - cross-validation,
   - and performance comparison using multiple metrics.

Evaluation metrics may include:
- Accuracy,
- Precision,
- Recall,
- F1-score,
- Confusion Matrices,
- ROC-AUC,
- and learning curves.

---

# Repository Structure

```text
├── data/               # Dataset loading and preprocessing
├── notebooks/          # Jupyter notebooks for experiments and analysis
├── src/                # Model training and utility scripts
├── results/            # Saved outputs, metrics, graphs, and evaluation results
│   └── figures/        # Visualisations used in the report
├── README.md
└── requirements.txt
```

---

# Team Contributions

This project is completed collaboratively by a group of four students. Responsibilities include:

- dataset preprocessing and exploratory analysis,
- feature engineering and validation,
- machine learning model implementation,
- neural network development,
- evaluation and experimentation,
- report writing and result interpretation.

---

# Disclaimer

This project is developed for academic purposes as part of COMP90049 Introduction to Machine Learning at the University of Melbourne.

Generative AI tools may only be used in accordance with the assignment policy for idea development, planning assistance, or short-phrase paraphrasing, and any usage will be formally declared in the final report.
