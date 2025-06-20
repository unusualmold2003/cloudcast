from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

app = Flask(__name__)

# Load and prepare data
def load_data():
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to find the CSV file
    csv_path = os.path.join(os.path.dirname(current_dir), 'Sub_Division_IMD_2017.csv')
    df = pd.read_csv(csv_path)
    return df

# Enhanced model training with trend analysis
def train_models():
    df = load_data()
    models = {}
    
    # Get all unique regions from the data
    regions = df['SUBDIVISION'].unique()
    
    # Filter for our target regions if they exist in data
    target_regions = ['Tamil Nadu', 'Coastal Karnataka', 'North Interior Karnataka', 
                     'South Interior Karnataka', 'Kerala']
    
    available_regions = [region for region in target_regions if region in regions]
    
    print(f"Training models for regions: {available_regions}")
    
    for region in available_regions:
        region_data = df[df['SUBDIVISION'] == region].copy()
        
        if len(region_data) > 0:
            print(f"Training {region}: {len(region_data)} years of data ({region_data['YEAR'].min()}-{region_data['YEAR'].max()})")
            
            months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                     'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            
            models[region] = {}
            
            for month in months:
                # Use year as feature
                X = region_data[['YEAR']].values
                y = region_data[month].values
                
                # Remove any NaN values
                mask = ~np.isnan(y) & ~np.isnan(X.flatten())
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) > 10:  # Need sufficient data points
                    # Create a pipeline with polynomial features for better trend capture
                    model = Pipeline([
                        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, 
                                                   max_depth=8, min_samples_split=5))
                    ])
                    
                    # Fit the model
                    model.fit(X_clean, y_clean)
                    models[region][month] = model
                    
                    # For years after 2023, use 2023 pattern
                    models[region][f'{month}_2023_value'] = region_data[region_data['YEAR'] == 2023][month].values[0] if len(region_data[region_data['YEAR'] == 2023]) > 0 else np.mean(y_clean[-5:])
                
                elif len(X_clean) > 0:
                    # If insufficient data for ML, use mean of available data
                    models[region][month] = np.mean(y_clean)
                    models[region][f'{month}_2023_value'] = np.mean(y_clean)
    
    return models

# Initialize models (this will run on cold start)
models = None

def get_models():
    global models
    if models is None:
        models = train_models()
    return models

@app.route('/')
def index():
    return send_from_directory('../static', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict_rainfall():
    try:
        models = get_models()
        data = request.json
        region = data['region']
        year = int(data['year'])
        
        # Validate year (accept any reasonable year)
        if year < 1800 or year > 2100:
            return jsonify({'error': 'Year must be between 1800 and 2100'}), 400
        
        # Map region names
        region_mapping = {
            'South Interior Karnataka': 'South Interior Karnataka',
            'North Interior Karnataka': 'North Interior Karnataka', 
            'Coastal Karnataka': 'Coastal Karnataka',
            'Tamil Nadu': 'Tamil Nadu',
            'Kerala': 'Kerala'
        }
        
        if region not in region_mapping:
            return jsonify({'error': 'Invalid region'}), 400
            
        mapped_region = region_mapping[region]
        
        if mapped_region not in models:
            return jsonify({'error': f'No model available for {mapped_region}. Available regions: {list(models.keys())}'}), 400
        
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        
        predictions = {}
        errors = {}
        
        # Get actual data if available
        df = load_data()
        actual_data = df[(df['SUBDIVISION'] == mapped_region) & (df['YEAR'] == year)]
        
        for month in months:
            if month in models[mapped_region]:
                if isinstance(models[mapped_region][month], (int, float)):
                    # Simple mean value
                    pred = models[mapped_region][month]
                else:
                    # ML model prediction
                    if year > 2023:
                        # For future years, use 2023 pattern with slight variation
                        base_2023 = models[mapped_region].get(f'{month}_2023_value', 
                                                            models[mapped_region][month].predict([[2023]])[0])
                        # Add small random variation for future years
                        variation = np.random.normal(0, 0.05) * base_2023  # 5% variation
                        pred = base_2023 + variation
                    else:
                        # Historical or recent prediction
                        pred = models[mapped_region][month].predict([[year]])[0]
                
                predictions[month] = max(0, round(pred, 1))  # Ensure non-negative
                
                # Calculate error if actual data exists
                if not actual_data.empty and not pd.isna(actual_data[month].iloc[0]):
                    actual = actual_data[month].iloc[0]
                    error = abs(pred - actual)
                    error_percentage = (error / max(actual, 1)) * 100  # Avoid division by zero
                    errors[month] = {
                        'absolute_error': round(error, 1),
                        'percentage_error': round(error_percentage, 1),
                        'actual': round(actual, 1)
                    }
                else:
                    errors[month] = {
                        'absolute_error': 'N/A',
                        'percentage_error': 'N/A', 
                        'actual': 'N/A'
                    }
            else:
                predictions[month] = 'N/A'
                errors[month] = {
                    'absolute_error': 'N/A',
                    'percentage_error': 'N/A',
                    'actual': 'N/A'
                }
        
        # Calculate total annual prediction
        valid_predictions = [v for v in predictions.values() if v != 'N/A']
        annual_pred = sum(valid_predictions) if valid_predictions else 0
        
        # Calculate overall error metrics
        if not actual_data.empty:
            actual_annual = actual_data['ANNUAL'].iloc[0] if not pd.isna(actual_data['ANNUAL'].iloc[0]) else None
            if actual_annual:
                annual_error = abs(annual_pred - actual_annual)
                annual_error_pct = (annual_error / actual_annual) * 100
                overall_error = {
                    'mae': round(annual_error, 1),
                    'percentage_error': round(annual_error_pct, 1),
                    'actual_annual': round(actual_annual, 1)
                }
            else:
                overall_error = {'mae': 'N/A', 'percentage_error': 'N/A', 'actual_annual': 'N/A'}
        else:
            overall_error = {'mae': 'N/A', 'percentage_error': 'N/A', 'actual_annual': 'N/A'}
        
        # Add prediction confidence indicator
        if year <= 2023:
            confidence = "High (Historical/Recent Data)"
        elif year <= 2030:
            confidence = "Medium (Near Future Projection)"
        else:
            confidence = "Low (Far Future Projection)"
        
        return jsonify({
            'region': region,
            'year': year,
            'predictions': predictions,
            'errors': errors,
            'annual_prediction': round(annual_pred, 1),
            'overall_error': overall_error,
            'confidence': confidence,
            'data_range': f"Model trained on years {df[df['SUBDIVISION'] == mapped_region]['YEAR'].min()}-{df[df['SUBDIVISION'] == mapped_region]['YEAR'].max()}"
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/regions')
def get_regions():
    models = get_models()
    # Return only regions that have trained models
    available_regions = list(models.keys())
    return jsonify(available_regions)

@app.route('/api/data-info')
def get_data_info():
    """Get information about available data"""
    models = get_models()
    df = load_data()
    info = {}
    
    for region in models.keys():
        region_data = df[df['SUBDIVISION'] == region]
        info[region] = {
            'years_available': f"{region_data['YEAR'].min()}-{region_data['YEAR'].max()}",
            'total_records': len(region_data),
            'sample_years': sorted(region_data['YEAR'].unique())[:10]  # First 10 years as sample
        }
    
    return jsonify(info)

# For Vercel serverless
app = app
