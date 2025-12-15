# 🎓 Predictive-Analytics-for-Student-Scores

## Project Overview

This project implements a foundational predictive analytics workflow on the **Student Performance dataset** (`StudentsPerformance.csv`). The goal is twofold:

1.  Perform comprehensive Exploratory Data Analysis (EDA) and visualization to understand the factors influencing student scores (Unit I).
2.  Develop and evaluate a **Simple Linear Regression** model to predict student Math Scores (Unit II).

The entire process is structured into a clear, 7-step Machine Learning pipeline for reproducibility.

### Core Concepts Covered:

  * Data Preprocessing and EDA (Unit I)
  * Visualizations and Correlation Analysis (Unit II)
  * Simple Linear Regression (Unit II)
  * Model Evaluation using MAE, MSE, R-squared (Unit II)

-----

## 🛠 Project Structure and Setup

### Prerequisites

You need the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Files in this Repository

| File | Description |
| :--- | :--- |
| `code.py` | The main script containing the full 7-step ML pipeline, comparison table output, and 7 enhanced visualizations. |
| `StudentsPerformance.csv` | The dataset used for modeling. |
| `README.md` | This file. |

-----

## 🚀 The 7-Step Machine Learning Pipeline

The project code executes the following steps sequentially:

### **1. Import Libraries**

  * Imports core libraries: `pandas`, `numpy`, `sklearn` (for modeling), `matplotlib`, and `seaborn` (for visualization).

### **2. Load the Dataset**

  * Loads the `StudentsPerformance.csv` file into a Pandas DataFrame.

### **3. Dataset Cleaning & EDA (Exploratory Data Analysis)**

  * **Cleaning:** Checks for missing data (confirmed no null values).
  * **Comparison Output:** Generates and prints a critical text table comparing **Mean Math Scores** by Parental Education Level and Gender.
  * **Visualizations:** Generates **7 enhanced visualizations** (e.g., Correlation Heatmap, KDE distributions, Grouped Bar Chart) to understand variable relationships and data distribution.

### **4. Data Splitting**

  * Selects **Reading Score** as the feature (`X`) and **Math Score** as the target (`y`).
  * Splits the data into 70% for training and 30% for testing.

### **5. Model Import**

  * Initializes the `LinearRegression` model from `sklearn`.

### **6. Predicting Values using Model**

  * Trains the model on the training data (`X_train`, `y_train`).
  * Generates predictions (`y_pred`) on the unseen test data (`X_test`).

### **7. Checking the Performance of Model**

  * Calculates and prints the key regression evaluation metrics from Unit II: **R-squared**, **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **Root Mean Squared Error (RMSE)**.

-----

## 📈 Key Findings from Visualization

The extensive EDA provided key insights before modeling:

| Visualization | Key Insight | Relevance |
| :--- | :--- | :--- |
| **Correlation Heatmap** | High correlation between all three scores (`math`, `reading`, `writing`), confirming that `reading score` is a strong predictor for `math score`. | **Unit II: Correlations** |
| **Comparison Table (Text/Viz)** | Students whose parents have Master's or Bachelor's degrees consistently achieve higher mean scores. A gender gap exists across all education levels. | **Unit I: Data Preparation/Feature Importance** |
| **KDE Distribution** | Students who completed the test preparation course show a clear shift towards higher scores compared to those who did not. | **Unit I: Feature Impact** |

-----

## 📊 Model Performance Results

The model's performance on the test set is evaluated against the Unit II metrics:

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **R-squared ($R^2$)** | $\text{[Insert R2 value from step 7]}$ | The model explains **[Insert R2 % value]** of the variance in math scores. |
| **Mean Absolute Error (MAE)** | $\text{[Insert MAE value from step 7]}$ | The average absolute error in prediction is approximately [Insert MAE value] points. |
| **Root Mean Squared Error (RMSE)** | $\text{[Insert RMSE value from step 7]}$ | The typical prediction error is [Insert RMSE value] points. |

-----

## Author

  * **[Chaithanya Sudheer Sakamuri]** - **[Data Science Student]**
  * *Course:* Predictive Analytics (INT234)
  * *Institution:* [Lovely Professinal University]
