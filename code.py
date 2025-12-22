# Google Play Store App Rating Prediction
# INT234 - Predictive Analytics CA2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

pd.set_option('future.no_silent_downcasting', True)

# Load the dataset
df = pd.read_csv(r"C:\Users\Dell\Downloads\googleplaystore.csv")

print("="*80)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

# Display basic information
print("\nFirst 5 rows of dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values in Dataset:")
print(df.isnull().sum())

# Data Cleaning
print("\n" + "="*80)
print("DATA CLEANING")
print("="*80)

# Remove rows where Rating is missing
df = df.dropna(subset=['Rating']).copy()
print("\nRemoved rows with missing Rating values")

# Clean Size column
print("\nCleaning Size column...")
df['Size'] = df['Size'].astype(str)
df['Size'] = df['Size'].str.replace('M', '', regex=False)
df['Size'] = df['Size'].str.replace('k', '', regex=False)
df['Size'] = df['Size'].replace('Varies with device', np.nan)
df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
df['Size'] = df['Size'].fillna(df['Size'].median())

# Clean Installs column
print("Cleaning Installs column...")
df['Installs'] = df['Installs'].astype(str)
df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
df['Installs'] = df['Installs'].fillna(df['Installs'].median())

# Clean Price column
print("Cleaning Price column...")
df['Price'] = df['Price'].astype(str)
df['Price'] = df['Price'].str.replace('$', '', regex=False)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Price'] = df['Price'].fillna(0)

# Clean Reviews column
print("Cleaning Reviews column...")
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df['Reviews'] = df['Reviews'].fillna(df['Reviews'].median())

print("\nData after cleaning:")
print(df.describe())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Data Visualization
print("\n" + "="*80)
print("DATA VISUALIZATION")
print("="*80)

# Plot 1: Distribution of Ratings
plt.figure(figsize=(8, 5))
plt.hist(df['Rating'], bins=20, color='lightblue', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Number of Apps')
plt.title('Distribution of App Ratings')
plt.grid(True, alpha=0.3)
plt.show()

# Plot 2: Top 10 Categories
plt.figure(figsize=(10, 6))
top_categories = df['Category'].value_counts().head(10)
plt.barh(top_categories.index, top_categories.values, color='coral')
plt.xlabel('Number of Apps')
plt.ylabel('Category')
plt.title('Top 10 App Categories')
plt.gca().invert_yaxis()
plt.show()

# Plot 3: Installs vs Rating
plt.figure(figsize=(8, 5))
plt.scatter(df['Installs'], df['Rating'], alpha=0.4, color='green')
plt.xscale('log')
plt.xlabel('Number of Installs')
plt.ylabel('Rating')
plt.title('Installs vs Rating')
plt.grid(True, alpha=0.3)
plt.show()

# Plot 4: Free vs Paid Apps
plt.figure(figsize=(7, 5))
df['Type'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.xlabel('App Type')
plt.ylabel('Count')
plt.title('Free vs Paid Apps Distribution')
plt.xticks(rotation=0)
plt.show()

# Create Target Variable
print("\n" + "="*80)
print("CREATING TARGET VARIABLE")
print("="*80)

# Create binary classification: High Rating (>=4.2) vs Low Rating (<4.2)
df['High_Rating'] = df['Rating'].apply(lambda x: 1 if x >= 4.2 else 0)

print("\nTarget Variable Distribution:")
print(df['High_Rating'].value_counts())
print("\n0 = Low Rating (Below 4.2)")
print("1 = High Rating (4.2 and above)")

# Feature Selection
print("\n" + "="*80)
print("FEATURE SELECTION")
print("="*80)

features = ['Category', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating']
print("\nSelected Features:", features)

X = df[features].copy()
y = df['High_Rating']

# Encoding Categorical Variables
print("\n" + "="*80)
print("ENCODING CATEGORICAL VARIABLES")
print("="*80)

cat_cols = ['Category', 'Type', 'Content Rating']
num_cols = ['Reviews', 'Size', 'Installs', 'Price']

le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    print(f"Encoded {col}")

# Scaling Numerical Features
print("\n" + "="*80)
print("SCALING NUMERICAL FEATURES")
print("="*80)

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
print("Numerical features scaled successfully")

# Train-Test Split
print("\n" + "="*80)
print("SPLITTING DATA INTO TRAIN AND TEST SETS")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Store accuracies for comparison
accuracy_scores = {}

# CLASSIFICATION MODELS
print("\n" + "="*80)
print("BUILDING AND EVALUATING CLASSIFICATION MODELS")
print("="*80)

# 1. K-Nearest Neighbors
print("\n" + "-"*80)
print("1. K-NEAREST NEIGHBORS (KNN)")
print("-"*80)

knn_classifier = KNeighborsClassifier(n_neighbors=9, weights='distance')
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)

knn_acc = accuracy_score(y_test, knn_predictions)
accuracy_scores['KNN'] = knn_acc

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, knn_predictions))
print(f"\nAccuracy: {knn_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, knn_predictions))

# 2. Support Vector Machine
print("\n" + "-"*80)
print("2. SUPPORT VECTOR MACHINE (SVM)")
print("-"*80)

svm_classifier = SVC(kernel='rbf', C=10, gamma='scale')
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)

svm_acc = accuracy_score(y_test, svm_predictions)
accuracy_scores['SVM'] = svm_acc

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))
print(f"\nAccuracy: {svm_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, svm_predictions))

# 3. Naive Bayes
print("\n" + "-"*80)
print("3. NAIVE BAYES")
print("-"*80)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)

nb_acc = accuracy_score(y_test, nb_predictions)
accuracy_scores['Naive Bayes'] = nb_acc

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, nb_predictions))
print(f"\nAccuracy: {nb_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, nb_predictions))

# 4. Decision Tree
print("\n" + "-"*80)
print("4. DECISION TREE")
print("-"*80)

dt_classifier = DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=42)
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)

dt_acc = accuracy_score(y_test, dt_predictions)
accuracy_scores['Decision Tree'] = dt_acc

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_predictions))
print(f"\nAccuracy: {dt_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, dt_predictions))

# Model Comparison
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

print("\nAccuracy Summary:")
print("-" * 40)
for model, acc in accuracy_scores.items():
    print(f"{model:20s}: {acc * 100:.2f}%")

# Visualization of Model Comparison
plt.figure(figsize=(8, 5))
models = list(accuracy_scores.keys())
accuracies = list(accuracy_scores.values())
colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow']

plt.bar(models, accuracies, color=colors, edgecolor='black')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Comparison of Classifier Accuracies')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v*100:.2f}%', ha='center', fontweight='bold')
plt.show()

# Final Recommendation
print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

best_classifier = max(accuracy_scores, key=accuracy_scores.get)
best_acc = accuracy_scores[best_classifier]

print(f"\nBest Performing Classifier: {best_classifier}")
print(f"Highest Accuracy: {best_acc * 100:.2f}%")
print(f"\nConclusion: The {best_classifier} classifier is the most suitable")
print("model for predicting high-rated Google Play Store applications")
print("based on the given dataset.")
print("="*80)
