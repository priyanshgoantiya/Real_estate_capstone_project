# Real_estate_capstone_project
Capstone Project Overview: Real Estate Data Science Application

This capstone project leverages data science techniques to provide insights, predictions, and recommendations in the real estate domain. The project includes stages of data gathering, cleaning, exploratory analysis, modeling, recommendation system development, and deployment of an interactive application.

Data Gathering:
The project began with scraping real estate data from the 99acres website, supplemented by datasets from other property listing platforms, ensuring a diverse and representative dataset.

Data Cleaning and Merging:
Data cleaning addressed missing values and ensured consistency across the dataset. The data was then merged, combining house and flat information into a single dataset for comprehensive analysis.

Feature Engineering:
New features were created, such as room indicators, area specifications, possession age, furnishing details, and a luxury score. These features enriched the dataset and improved property representation.

Exploratory Data Analysis (EDA):
Univariate and multivariate analyses revealed patterns and relationships within the data. Tools like Pandas Profiling helped uncover data distribution and structure insights.

Outlier Detection and Missing Value Imputation:
Outliers were removed, and missing values in critical columns, such as area and bedrooms, were addressed through appropriate imputation techniques.

Feature Selection:
Techniques like correlation analysis, random forest, gradient boosting, LASSO, and SHAP were used to identify key variables for modeling.

Model Selection & Productionalization:
Regression models, including Linear Regression, SVR, Random Forest, and MLP, were compared to predict property prices. The best-performing model was selected and deployed using Streamlit, offering an interactive web interface for users.

Analytics Module:
A visualization module was built to display insights through maps, plots, and statistical graphics, helping users understand the real estate market.

Recommender System:
Three recommendation models were developed, focusing on facilities, price, and location advantages, all accessible through an intuitive Streamlit interface.

Deployment on AWS:
The application was deployed on AWS to ensure scalability and accessibility, providing users with predictions, analytics, and recommendations.

This project demonstrates proficiency in data science and the deployment of a real-world application offering valuable real estate insights to users.
