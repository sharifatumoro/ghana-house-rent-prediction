# Ghana House Rent Prediction (Machine Learning Project)

## Overview

This repository contains an **end-to-end machine learning project for predicting house rental prices in Ghana**. The project applies **data science, econometrics, and regression-based machine learning techniques** to analyze real estate rental data and generate accurate rent predictions.

It is designed as a **portfolio-quality project** suitable for:

* Data Science & Machine Learning roles
* Economics & Applied Econometrics research
* Academic projects and applied analytics


## Key Features

* Real-world **Ghana housing rental dataset**
* Full **data science pipeline** (cleaning → EDA → modeling → evaluation)
* Feature engineering with **scaling, categorical and target mean encoders**
* Regression models including **Linear, Lasso, Ridge and Elastic Regression**
* Reproducible notebooks and deployment-ready structure


## Tech Stack

* **Programming:** Python
* **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, statmodels
* **Modeling:** Linear Regression, Ridge Regression, Lasso Regression, Elastic Regression
* **Tools:** Jupyter Notebook, Git, GitHub


##  Project Structure
```
GHANA_HOUSE_RENT_PROJECT/
│
├── app/
│   └── app.py                  # Inference / prediction script
│
├── data/
|   ├── house_rentals.csv
│   ├── cleaned_house_rentals.csv
│   ├── processed_house_rentals.csv
│   
├── model/                      # Saved model artifacts (ignored in GitHub)
│   ├──final_model.pkl
|
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── eda_visualization.ipynb
│   ├── preprocessing.ipynb
│   └── training&evaluation.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```


## Workflow

1. **Data Cleaning:** handled missing values, outliers, and inconsistencies
2. **Exploratory Data Analysis (EDA):** rent distributions and feature relationships
3. **Feature Engineering:** scaling, ordinal encoding, target mean encoding
4. **Model Training:** regression models with regularization
5. **Evaluation:** MAE, RMSE, and R² comparison


## Results & Insights

* Ridge Regression improved generalization and reduced overfitting
* Amenities, locality and bedrooms were the strongest predictors of rent
* The final model achieved stable and interpretable performance


## ▶️ How to Run Locally

```bash
git clone https://github.com/your-username/ghana-house-rent-prediction.git
cd ghana-house-rent-prediction
pip install -r requirements.txt
python app/app.py
```



## Author

**Sharifatu Moro**
Aspiring Data Scientist & Machine Learning Engineer
Background: Economics & Applied Data Analysis


## GitHub Topics

```
data-science
machine-learning
real-estate
regression
economics
ghana
housing-market
```


If you find this project useful, please consider starring the repository.
