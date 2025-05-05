# Airbnb Price Prediction Project

## Project Overview

This machine learning project aims to predict Airbnb listing prices and identify key factors influencing pricing decisions. The system helps property owners optimize their rental prices using data-driven insights while remaining competitive in the market.

## Key Features

- **Predictive Modeling**: XGBoost regression model to estimate listing prices
- **Feature Analysis**: Identifies top price influencers (room type, location, amenities)
- **Data Processing**: Handles missing values, feature engineering, and categorical encoding
- **Business Insights**: Actionable recommendations for hosts and Airbnb platform

## Project Structure

```
airbnb-price-prediction/
├── data/                    # Dataset files
│   └── Airbnb_data.csv
├── notebooks/               # Jupyter notebooks
│   ├── 1_airbnb_price_prediction-using-ML
├── models/                  # Saved models
├── gradio-app/              # Project documentation
│   └── app.py
└── README.md
```

## Dataset Information

The dataset contains 74,111 listings with 29 features including:

- Property features (type, bedrooms, bathrooms)
- Location data (city, latitude/longitude)
- Host details (response rate, verification status)
- Amenities (WiFi, kitchen, etc.)
- Target variable: `log_price` (log-transformed price)

## Methodology

1. **Data Preprocessing**:
   - Handled missing values with median imputation
   - Processed amenities from JSON-like strings to count features
   - Created temporal features (host duration, review period)
   - Encoded categorical variables (one-hot and binary encoding)

2. **Model Development**:
   - XGBoost Regressor with hyperparameter tuning
   - Feature selection (22 key features)
   - Pipeline with preprocessing and regression

3. **Evaluation Metrics**:
   - RMSE: 0.383
   - R²: 0.72

## Key Findings

- **Top Price Influencers**:
  1. Room Type (Entire home/apt has highest impact)
  2. Location (city and zipcode)
  3. Number of bedrooms and bathrooms

- **Business Recommendations**:
  - List as "Entire home" for higher value
  - Highlight key amenities (WiFi, kitchen)
  - Optimize location tags in listings



## Future Work

- Incorporate image analysis and NLP for listing descriptions
- Geospatial analysis for proximity to attractions
- Web app deployment for real-time predictions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

