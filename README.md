# Smart Property Price Predictor

## Description
Smart Property Price Predictor is an interactive web app that leverages regression and classification machine learning models to forecast property values five years into the future and evaluate investment opportunities. Users can input property details to receive predictions and recommendations, making it a valuable tool for real estate analysis.

## Features
- **Price Prediction**: Estimate property value after 5 years using current details.
- **Investment Assessment**: Determine if a property is a good investment with probability scores.
- **User-Friendly Interface**: Intuitive inputs for property attributes, including amenities and location.
- **Feature Importance Visualization**: See which factors most influence predictions.
- **Dataset Insights**: Explore statistics and visualizations of the underlying data.
- **Model Flexibility**: Supports multiple ML algorithms for robust predictions.

## Prerequisites
- Python 3.7 or higher
- Internet connection for downloading dependencies and data

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd smart-property-price-predictor
   ```
2. Install required packages:
   ```
   pip install streamlit pandas joblib numpy scikit-learn xgboost matplotlib seaborn
   ```

## Data and Models
Due to file size constraints, the dataset and trained models are not included in the repository. Follow these steps to obtain them:

### Dataset
- Download `india_housing_prices.csv` from sources like Kaggle or the original provider.
- Alternatively, use `cleaning.ipynb` to process raw data into `cleaned_dataset.csv`.

### Trained Models
- Execute `model.ipynb` to train and save models (e.g., `.pkl` files).
- Or download pre-trained models from a provided cloud link.

Place all files in the project root directory.

## Usage
1. Launch the app:
   ```
   streamlit run app.py
   ```
2. Access the app at `http://localhost:8501`.
3. Enter property details (city, BHK, size, etc.).
4. Click "Predict" to view results: future price, investment recommendation, and probability.

## Models Overview
The app employs pre-trained models for prediction tasks:

### Regression Models (Price Prediction)
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

### Classification Models (Investment Assessment)
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Currently, Random Forest models are active in the interface.

## Dataset Description
- **Raw Data**: `india_housing_prices.csv` – Original Indian property listings.
- **Processed Data**: `cleaned_dataset.csv` – Cleaned dataset for training.
- **Features**: City, BHK, size (sq ft), year built, property type, furnished status, amenities, parking, security.
- **Targets**:
  - Regression: Projected price after 5 years (assuming 5% annual growth).
  - Classification: Investment quality (based on price per sq ft vs. median).

## Project Structure
```
smart-property-price-predictor/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── cleaning.ipynb            # Data cleaning notebook
├── EDA_data.ipynb            # Exploratory data analysis
├── model.ipynb               # Model training notebook
├── cleaned_dataset.csv       # Processed dataset
├── india_housing_prices.csv  # Raw dataset
├── *.pkl                     # Trained model files
└── README.md                 # This file
```


## Author
SHAZIA AFRAZ
