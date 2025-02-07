import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load and prepare the data
def load_data():
    df = pd.read_csv('data/monthly_report_expanded.csv')
    return df

def prepare_features(df):
    # Separate last month and month before last metrics
    last_month = df[[col for col in df.columns if 'last month' in col]]
    month_before = df[[col for col in df.columns if 'month before last' in col]]
    
    # Clean column names
    last_month.columns = [col.replace(' (last month)', '') for col in last_month.columns]
    month_before.columns = [col.replace(' (month before last)', '') for col in month_before.columns]
    
    return last_month, month_before

def analyze_performance_metrics():
    df = load_data()
    last_month, month_before = prepare_features(df)
    
    # Calculate key performance indicators
    metrics = {
        'conversion_rate': last_month['Demo Meeting Completed'] / last_month['Demo Meeting Set'],
        'opportunity_creation_rate': last_month['Opportunity Created'] / last_month['Demo Meeting Completed'],
        'activity_efficiency': last_month['Demo Meeting Set'] / (last_month['Outbound Calls'] + last_month['Personalized Outbound Emails'])
    }
    
    return metrics

def build_predictive_models(df):
    # Prepare features for opportunity creation prediction
    X = df[[
        'Outbound Calls', 'Personalized Outbound Emails', 
        'Calls with Correct Contact', 'Demo Meeting Set',
        'Demo Meeting Completed', 'Personal Development Hour'
    ]]
    y = df['Opportunity Created']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = xgb.XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, feature_importance

def analyze_time_patterns():
    df = load_data()
    last_month, month_before = prepare_features(df)
    
    # Compare activities between months
    activity_comparison = pd.DataFrame({
        'last_month': last_month.mean(),
        'month_before': month_before.mean(),
        'change': (last_month.mean() - month_before.mean()) / month_before.mean() * 100
    })
    
    return activity_comparison

def generate_recommendations(metrics, feature_importance, activity_comparison):
    recommendations = {
        'high_impact_activities': feature_importance.head(3)['feature'].tolist(),
        'improvement_areas': activity_comparison[activity_comparison['change'] < 0].index.tolist(),
        'optimal_targets': {
            'daily_calls': int(metrics['activity_efficiency'].mean() * 100),
            'email_personalization_rate': 0.8,
            'development_hours': 5
        }
    }
    
    return recommendations

if __name__ == "__main__":
    # Execute analysis pipeline
    df = load_data()
    metrics = analyze_performance_metrics()
    model, feature_importance = build_predictive_models(df)
    activity_patterns = analyze_time_patterns()
    recommendations = generate_recommendations(metrics, feature_importance, activity_patterns)

