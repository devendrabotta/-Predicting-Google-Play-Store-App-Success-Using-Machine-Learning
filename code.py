#import library
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

#warning suspension
warnings.filterwarnings("ignore", category=FutureWarning)

#load the data set
df = pd.read_csv('StudentsPerformance.csv')
education_order = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
print("-" * 70)

#dataset cleaning eda 
#visuvalization
sns.set_theme(style="darkgrid")
plt.ioff()

#comparison eda
mean_scores_comparison = df.groupby(['parental level of education', 'gender'])['math score'].mean().unstack()
mean_scores_comparison = mean_scores_comparison.reindex(education_order).round(2)
print("\n--- Comparison Table (EDA Output) ---")
print(mean_scores_comparison)
print("\nMissing values check (Data Cleaning):")
print(df.isnull().sum()) 

#feature selection for modeling 
X = df[['reading score']].values.reshape(-1, 1) # Feature (Reading Score)
y = df['math score'].values                    # Target (Math Score)
print("\nFeatures (X) and Target (y) selected for Regression.")

#bar chart count - test preparation  course bar chart count
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, y='test preparation course', order=df['test preparation course'].value_counts().index, palette='crest')
plt.title('1. Count of Test Preparation Course Completion (Unit I)', fontsize=12)
plt.xlabel('Count', fontsize=10)
plt.ylabel('Test Preparation Course', fontsize=10)
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.), ha='left', va='center', fontsize=9)
plt.show()

#kde distribution - mathscore by test prep 
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x='math score', hue='test preparation course', fill=True, alpha=.5, linewidth=2)
plt.title('2. Distribution of Math Scores by Test Prep (Unit II)', fontsize=12)
plt.xlabel('Math Score', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.legend(title='Test Prep', labels=['Completed', 'None'])
plt.show()

#box plot category vs numerical - math score by test prep
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='test preparation course', y='math score', palette='Set2', linewidth=1.5)
plt.title('3. Math Score Distribution by Test Prep Course', fontsize=12)
plt.xlabel('Test Preparation Course', fontsize=10)
plt.ylabel('Math Score', fontsize=10)
plt.show()

#scatter Plot - math vs reading score
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='math score', y='reading score', hue='gender', style='gender', alpha=0.7, s=50)
plt.title('4. Math Score vs Reading Score (Correlation)', fontsize=12)
plt.xlabel('Math Score', fontsize=10)
plt.ylabel('Reading Score', fontsize=10)
plt.legend(title='Gender', fontsize=8, loc='upper left')
plt.show()

#stacked bar chart - parental education vs lunch
plt.figure(figsize=(8, 5))
lunch_counts_plot = df.groupby('parental level of education')['lunch'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
lunch_counts_plot = lunch_counts_plot.reindex(education_order)
lunch_counts_plot.plot(kind='bar', stacked=True, colormap='tab10', legend=False)
plt.title('5. Lunch Type Distribution by Parental Education Level', fontsize=12)
plt.xlabel('Parental Level of Education', fontsize=10)
plt.ylabel('Percentage (%)', fontsize=10)
plt.xticks(rotation=40, ha='right', fontsize=9)
plt.legend(title='Lunch Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#correlation heatmap - numerical scores
plt.figure(figsize=(6, 5))
corr_matrix = df[['math score', 'reading score', 'writing score']].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='mako', cbar_kws={'shrink': 0.8})
plt.title('6. Correlation Heatmap of Student Scores', fontsize=12)
plt.show()

#comparison plot - grouped bar chart
plt.figure(figsize=(8, 5))
ax = mean_scores_comparison.plot(kind='bar', colormap='coolwarm')
plt.title('7. Comparison: Mean Math Score by Parental Education and Gender', fontsize=12)
plt.xlabel('Parental Level of Education', fontsize=10)
plt.ylabel('Mean Math Score', fontsize=10)
plt.xticks(rotation=40, ha='right', fontsize=9)
plt.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', fontsize=8)
plt.tight_layout()
plt.show()

#dataspliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("-" * 70)
print("dataspliting (30% test size):")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

#model import
model = LinearRegression()
print("-" * 70)

#predicting values using model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("-" * 70)

#checking the performance of model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

#perfomance metrics
print("-" * 70)
print("checking the performance of model (Unit II Metrics):")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("-" * 70)
