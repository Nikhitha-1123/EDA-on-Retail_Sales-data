# 🛍️ Exploratory Data Analysis on Retail Sales

> ✅ Level 1- Task 1

This project performs **Exploratory Data Analysis (EDA)** on a retail sales dataset using Python. The goal is to uncover key trends, patterns, and insights to better understand product performance, sales trends, and customer behavior.

### 📂 Datasets Used
-`retail_sales_dataset.csv`: Contains transaction data (date, item, category, quantity sold, total sales)

-`menu.csv`: Includes item metadata (category, serving size, price, and nutritional info)

### 📊 Key Objectives
* Analyze sales trends over time (Time Series Analysis)
* Understand product-wise and category-wise performance
* Perform customer segmentation and behavior analysis
* Identify seasonal patterns and anomalies
* Visualize findings using matplotlib, seaborn, and plotly

### 🧪 Technologies Used
* Python
* Pandas
* NumPy
* Matplotlib & Seaborn
* Plotly
* Jupyter Notebook


### 📈 Sample Visualizations
* Bar Charts for top-selling categories/products
* Line Graphs for monthly/weekly sales
* Heatmaps for correlations and seasonal trends
* Pie Charts for product distribution

### 📌 Insights & Recommendations
* Certain product categories dominate revenue during holiday seasons
* Low-performing items identified for potential discontinuation
* Suggestions for inventory restocking and seasonal promotions


# 🧠 Customer Segmentation Analysis

> ✅ Level 1- Task 2

This project focuses on segmenting customers based on their demographics and purchasing behavior using machine learning techniques, primarily **KMeans clustering**.

### 📂 Dataset Used
- `ifood_df.csv`: Contains customer data including demographics, purchase amounts, and web behavior.

### 🔍 Key Concepts and Challenges Addressed
- **Data Collection**: Loaded customer and transaction data from a CSV file.
- **Data Exploration and Cleaning**: Handled missing values and understood data structure using `.info()` and `.describe()`.
- **Descriptive Statistics**: Explored distributions for income, age, and spending patterns.
- **Customer Segmentation**: Applied KMeans clustering after feature selection and standardization.
- **Visualization**: Used PCA, scatter plots, heatmaps, and boxplots for visual exploration.
- **Insights and Recommendations**: Analyzed cluster characteristics for strategic business targeting.

### 🧪 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

### 📊 Result
Successfully segmented customers into meaningful groups using KMeans clustering. Provided actionable insights for marketing and targeting based on income, age, and purchase behaviors.



# 🧠 Sentiment Analysis using NLP and Machine Learning

> ✅ Level 1- Task 4

This project focuses on analyzing the emotional tone of text data—positive, negative, or neutral—using Natural Language Processing (NLP) and machine learning models.

### 🔍 Objective
To classify sentiment in user-generated content from Twitter and app reviews using text preprocessing, feature engineering, and classification algorithms.

### 📂 Datasets Used
- `Twitter_Data.csv`: Cleaned tweets labeled with sentiment (-1, 0, 1)
- `user_reviews.csv`: Translated app reviews with sentiment labels (Positive, Neutral, Negative)

### 📌 Key Concepts Covered
- ✅ Sentiment Analysis
- ✅ Natural Language Processing (stopword removal, lemmatization)
- ✅ Feature Engineering (TF-IDF)
- ✅ Classification using Naive Bayes and SVM
- ✅ Data Visualization of sentiment distribution

### ⚙️ Technologies Used
- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- Matplotlib / Seaborn

### 📊 Models Implemented
- Multinomial Naive Bayes
- Support Vector Machine (SVM)

### 📈 Results
- Visualizations showing the distribution of sentiments
- Accuracy and classification reports of both models

### ✅ Conclusion
Successfully built a sentiment analysis pipeline capable of processing raw text data and classifying emotions with solid accuracy and insights.






# 🏡 Housing Price Prediction using Linear Regression

> ✅ Level 2- Task 1

This project demonstrates a complete machine learning pipeline to predict housing prices based on various features. It includes data cleaning, feature engineering, model training, evaluation, and visualization using Python and Scikit-Learn.



### 📂 Dataset Used

- `Housing.csv`: Contains numerical and categorical features related to residential properties and their selling prices.



### 🚀 Key Concepts & Workflow

1. **Data Collection**  
   Obtained a dataset with relevant numerical and categorical features for housing prices.

2. **Data Exploration and Cleaning**  
   - Handled missing values  
   - Converted categorical data using one-hot encoding  
   - Explored dataset structure and statistics

3. **Feature Selection**  
   - Selected features that influence house prices  
   - Dropped unnecessary columns

4. **Model Training**  
   - Implemented **Linear Regression** using `scikit-learn`  
   - Trained on 80% of the data

5. **Model Evaluation**  
   - Evaluated using **Mean Squared Error (MSE)** and **R² Score**

6. **Visualization**  
   - Plotted **Actual vs Predicted** house prices to visually assess model performance



### 🧠 Libraries Used

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn



### 📊 Evaluation Metrics

- **Mean Squared Error (MSE)**  
- **R-squared (R²) Score**


### 📌 Results

The model successfully predicted housing prices with reasonable accuracy using linear regression. The performance was evaluated with standard regression metrics and illustrated visually.






# 🍷 Wine Quality Prediction using Machine Learning

> ✅ Level 2- Task 2

This project focuses on predicting the quality of red wine based on physicochemical attributes using multiple classification models. The goal is to analyze key chemical properties and apply machine learning classifiers to predict wine quality effectively.

### 📂 Dataset Used

- `WineQT.csv`: Contains chemical properties of red wines and corresponding quality scores (ranging from 3 to 8).


### 📌 Key Concepts Covered

- **Classifier Models**: Random Forest, Stochastic Gradient Descent (SGD), Support Vector Classifier (SVC)
- **Chemical Analysis**: Focus on density, acidity, alcohol, sulphates, etc.
- **Data Handling**: Pandas and NumPy for cleaning and manipulation
- **Visualization**: Correlation heatmaps, boxplots, and confusion matrices using Seaborn and Matplotlib


### 🛠️ Project Workflow

1. Data Loading & Exploration
2. Data Cleaning and Feature Selection
3. Exploratory Data Analysis (EDA)
4. Feature Scaling using StandardScaler
5. Train-Test Split
6. Model Training:
   - Random Forest Classifier
   - SGD Classifier
   - SVC (Support Vector Classifier)
7. Model Evaluation:
   - Classification Report
   - Confusion Matrix
   - Feature Importance
   - Cross-Validation (Optional)



### 📈 Model Evaluation

Each model was evaluated using classification metrics such as:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix



### 🔍 Feature Importance

The Random Forest classifier revealed that `alcohol`, `sulphates`, and `volatile acidity` were among the most influential factors in predicting wine quality.



### 📌 Requirements

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn


# 💳 Credit Card Fraud Detection Project

>✅ Level 2- Task 3

A comprehensive machine learning project to detect fraudulent credit card transactions using anomaly detection and supervised classification models.


### 🚀 Project Overview

This project explores a real-world credit card dataset and applies both **unsupervised** and **supervised** machine learning techniques to detect fraud. It covers key concepts including anomaly detection, real-time monitoring, and system scalability.



### 🧠 Key Concepts Implemented

- 🔍 **Anomaly Detection**: Isolation Forest for unsupervised outlier detection.
- 🤖 **Machine Learning Models**: Random Forest classifier for supervised prediction.
- ⚙️ **Feature Engineering**: Scaling and transformation of time and amount features.
- 🛰️ **Real-time Monitoring**: Simulated fraud prediction in real-time.
- 📈 **Scalability Planning**: Strategy for handling large-scale production systems.


### 📂 Dataset

-`creditcard.csv`: Contains 284,807 European credit card transactions with anonymized features and labels indicating fraud (1) or legitimate (0)


### 📊 Model Performance

- **Anomaly Detection (Isolation Forest)**:
  - Precision & Recall compared against true labels
- **Supervised ML (Random Forest)**:
  - Confusion matrix, classification report
  - ROC-AUC: ~0.99



### 🧪 Tech Stack

- Python (Pandas, NumPy, Scikit-learn)
- Matplotlib & Seaborn for Visualization
- Jupyter Notebook


### 📈 Future Enhancements

- Add XGBoost / Neural Networks for comparison
- Save and deploy model via Flask API
- Stream data using Kafka for true real-time scoring
- Set up dashboard for live fraud alerts





# 📱 Google Play Store App Analysis

> ✅ Level 2- Task 4

This project involves data cleaning, exploratory data analysis (EDA), and sentiment analysis on Google Play Store data.

### 📂 Datasets Used
- `apps.csv`: Contains metadata of apps (category, installs, rating, price, etc.)
- `user_reviews.csv`: User reviews with sentiments (positive, negative, neutral)

### 🎯 Objectives
- Clean and prepare data for analysis
- Explore app categories and trends
- Analyze metrics like rating, installs, price
- Perform sentiment analysis on reviews
- Create interactive visualizations using Plotly

### 🧠 What I Learned
- Data preprocessing and handling real-world datasets
- Exploratory analysis using pandas, matplotlib, seaborn
- Interactive plotting with Plotly
- Sentiment labeling and review analysis
- Gained practical skills in storytelling with data

### 📊 Visuals
All visualizations are included in the Jupyter Notebook, including interactive charts.

### 🚀 Tools Used
- Python
- Pandas
- Matplotlib & Seaborn
- Plotly
- Jupyter Notebook





# 📬 Contact

For questions or collaboration, reach out to:

**Nikhitha**

📧 nikhithachallagonda@gmail.com

🌐 https://www.linkedin.com/in/nikhitha-challagonda
