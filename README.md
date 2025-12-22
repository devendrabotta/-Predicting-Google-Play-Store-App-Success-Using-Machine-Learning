📱 Google Play Store App Rating Prediction
📌 Project Overview

This project focuses on predicting the ratings of Google Play Store applications using machine learning techniques. By analyzing various app features such as category, reviews, size, installs, price, and content rating, the model aims to estimate an app’s rating accurately.

The project demonstrates the complete data science pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and evaluation.

🎯 Objective

To analyze factors affecting app ratings on the Google Play Store

To build a machine learning model that predicts app ratings

To gain insights into what makes an app successful

📊 Dataset Description

Source: Google Play Store dataset (public dataset)

Key Features:

App Category

Number of Reviews

App Size

Number of Installs

Price

Content Rating

Android Version

App Rating (Target Variable)

🛠️ Technologies Used

Programming Language: Python

Libraries:

pandas – data manipulation

numpy – numerical operations

matplotlib, seaborn – data visualization

scikit-learn – machine learning models

🔍 Exploratory Data Analysis (EDA)

Handling missing and inconsistent values

Removing outliers

Visualizing rating distributions

Analyzing relationships between features and ratings

Correlation analysis

⚙️ Data Preprocessing

Converted categorical variables using encoding techniques

Scaled numerical features

Removed irrelevant columns

Split data into training and testing sets

🤖 Machine Learning Model

Model Used: Regression Models (e.g., Linear Regression / Random Forest Regression)

Training: 80% of data

Testing: 20% of data

📈 Model Evaluation

The model performance is evaluated using:

R² Score

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

✅ Results

The model successfully predicts app ratings with reasonable accuracy

Features such as reviews, installs, and app category have a strong influence on ratings

🚀 Applications

Helps developers improve app quality

Assists businesses in app market analysis

Useful for recommendation and ranking systems

🔮 Future Enhancements

Use advanced models like XGBoost or Neural Networks

Include user sentiment analysis from app reviews

Deploy the model using Flask or Streamlit

📂 Project Structure
Google-Play-Store-App-Rating-Prediction/
│
├── dataset/
│   └── googleplaystore.csv
│
├── notebooks/
│   └── app_rating_prediction.ipynb
│
├── README.md
└── requirements.txt

👨‍💻 Author

Botta Devendra
B.Tech CSE | Data Science Enthusiast
