# 1: 🩺 Medical Insurance Price Prediction

This project analyzes medical insurance data to predict charges using regression techniques.

## 📊 Dataset
The dataset is from Kaggle:
[Medical Insurance Price Prediction](https://www.kaggle.com/datasets/harishkumardatalab/medical-insurance-price-prediction)

## 🧠 Steps Performed
- Data loading and inspection  
- Data cleaning and encoding categorical values  
- Exploratory data analysis (EDA) using Matplotlib and Seaborn  
- Building regression models for prediction  

## 🧩 Libraries Used
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
#
#

# 2: 🎲 Expert Backgammon Move Predictor using Machine Learning & Deep Learning

Welcome to the **Backgammon AI Engine**, an advanced machine learning and deep learning project designed to evaluate and predict the optimal expert moves for any valid game state, utilizing professional-level gameplay logs from competitive matches on Arkadium.

---

## 🚀 Project Overview

This system ingests historical backgammon match logs, parses turn-by-turn positions, auto-generates legal move sets using a deterministic simulator, and optimizes predictive models to match professional human behavior. We implement and benchmark three distinct systems:
1. **Heuristic Rule-Based Baseline:** Integrates expert strategies, giving high weight to checker-hitting priorities and bearing-off progress.
2. **XGBoost Classifier:** A gradient-boosted tree architecture utilizing flattened board coordinates, active player context, and dice values.
3. **Proposed Multi-Layer Perceptron (MLP):** A deep feedforward neural network engineered in PyTorch to generalize across non-linear relations on the board.

---

## 📋 Core ML Pipeline Steps

To construct this predictor, we proceed through the following structured engineering steps:
1. **Data Download & Ingestion:** Pull compressed tournament logs (`.mat` text files) programmatically using `huggingface_hub` and extract game sequences.
2. **Log Parsing:** Execute custom regex tokens to split games, extract dice values, verify moves, and record final match results.
3. **Legal Move Simulation:** Program a Backgammon logic engine that outputs all mathematically legal move sequences for any given state and dice combination.
4. **Target Label Encoding:** Align the human expert's chosen move with the index of the generated legal moves list (`chosen_idx`), saving the cleaned and fully parsed DataFrame as a serialized pickle file.
5. **Feature Engineering:** Flatten the 28-dimensional board state, append active player indices, and incorporate the physical dice values into a 31-dimensional dense vector.
6. **Data Splitting:** Stratify and divide data into Train (70%), Validation (15%), and Test (15%) splits to preserve stable class coverage across model variants.
7. **Model Training & Optimization:** Train an XGBoost model on GPU using histogram trees, and train our multi-layer PyTorch neural network optimizing cross-entropy loss with the Adam optimizer.
8. **Evaluation Metrics:** Apply strict masking rules on the models' raw probability distributions to screen out illegal moves, calculating **Top-3 Accuracy** and **Mean Reciprocal Rank (MRR)**.

---

## 📈 Performance Benchmarks

Our models are evaluated using rigorous metrics on completely unseen test data splits:

| Model | Validation Top-3 Acc | Validation MRR | Test Top-3 Acc | Test MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Heuristic Baseline** | ~38.4% | ~0.286 | ~38.5% | ~0.287 |
| **XGBoost Classifier** | ~54.3% | ~0.484 | ~54.3% | ~0.484 |
| **PyTorch MLP** | **~57.0%** | **~0.509** | **~57.3%** | **~0.510** |

---

## 🛠️ Tech Stack & Requirements

- **Languages & Frameworks:** Python 3.12, PyTorch, XGBoost, Scikit-Learn
- **Data Pipelines:** HuggingFace Hub (`huggingface_hub`), Pandas, NumPy, SciPy, Tqdm
