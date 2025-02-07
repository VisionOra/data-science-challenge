import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class SalesPredictor:
    def __init__(self, data_path='data/monthly_report_expanded.csv'):
        self.df = pd.read_csv(data_path)
        self.last_month, self.month_before = self._prepare_data()
        self.scaler = StandardScaler()
        
    def _prepare_data(self):
        """Prepare and split the data into last month and month before last"""
        last_month = self.df[[col for col in self.df.columns if 'last month' in col]]
        month_before = self.df[[col for col in self.df.columns if 'month before last' in col]]
        
        # Clean column names
        last_month.columns = [col.replace(' (last month)', '') for col in last_month.columns]
        month_before.columns = [col.replace(' (month before last)', '') for col in month_before.columns]
        
        return last_month, month_before

    def forecast_opportunity_creation(self):
        """Forecast next month's opportunity creation rate"""
        # Prepare features
        features = [
            'Outbound Calls', 'Personalized Outbound Emails',
            'Calls with Correct Contact', 'Demo Meeting Set',
            'Demo Meeting Completed', 'Personal Development Hour'
        ]
        
        X = pd.concat([self.last_month[features], self.month_before[features]], axis=1)
        y = self.last_month['Opportunity Created']
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        predictions = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'mse': mse,
            'feature_importance': importance,
            'predictions': predictions
        }

    def build_lead_scoring_model(self):
        """Build a lead scoring model to prioritize prospects"""
        # Create lead score based on conversion metrics
        self.last_month['lead_score'] = (
            (self.last_month['Demo Meeting Completed'] / self.last_month['Demo Meeting Set']) * 0.4 +
            (self.last_month['Opportunity Created'] / (self.last_month['Demo Meeting Completed'] + 1)) * 0.6
        )
        
        # Create binary classification target (high-value vs low-value leads)
        median_score = self.last_month['lead_score'].median()
        self.last_month['high_value_lead'] = (self.last_month['lead_score'] > median_score).astype(int)
        
        # Prepare features
        features = [
            'Outbound Calls', 'Personalized Outbound Emails',
            'Calls with Correct Contact', 'Custom LinkedIn Outreach to Candidates'
        ]
        
        X = self.last_month[features]
        y = self.last_month['high_value_lead']
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(model, X, y, cv=5)
        
        model.fit(X, y)
        
        return {
            'model': model,
            'cv_scores': scores,
            'feature_importance': pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        }

    def identify_at_risk_reps(self):
        """Identify at-risk sales reps before performance declines"""
        # Calculate performance metrics
        self.last_month['performance_score'] = (
            self.last_month['Opportunity Created'] * 0.4 +
            self.last_month['Demo Meeting Completed'] * 0.3 +
            self.last_month['Demo Meeting Set'] * 0.2 +
            self.last_month['Calls with Correct Contact'] * 0.1
        )
        
        # Calculate performance decline
        performance_decline = (
            self.last_month['performance_score'] - 
            self.month_before['performance_score']
        ) / self.month_before['performance_score']
        
        # Identify at-risk reps (those with declining performance)
        at_risk_reps = self.df[performance_decline < -0.1]['user_name'].tolist()
        
        # Calculate risk factors
        risk_factors = {
            'low_activity': self.last_month['Outbound Calls'] < self.last_month['Outbound Calls'].mean() * 0.7,
            'low_conversion': self.last_month['Demo Meeting Completed'] / self.last_month['Demo Meeting Set'] < 0.3,
            'declining_pipeline': performance_decline < -0.1
        }
        
        return {
            'at_risk_reps': at_risk_reps,
            'risk_factors': risk_factors,
            'performance_decline': performance_decline
        }

    def visualize_results(self, results_dict, plot_type):
        """Visualize analysis results"""
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'feature_importance':
            sns.barplot(
                x='importance',
                y='feature',
                data=results_dict['feature_importance']
            )
            plt.title('Feature Importance in Prediction Model')
            
        elif plot_type == 'performance_decline':
            sns.histplot(results_dict['performance_decline'])
            plt.title('Distribution of Performance Decline')
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Initialize predictor
    predictor = SalesPredictor()
    
    # Run all analyses
    opportunity_forecast = predictor.forecast_opportunity_creation()
    lead_scoring = predictor.build_lead_scoring_model()
    at_risk_analysis = predictor.identify_at_risk_reps()
    
    # Print key results
    print("\n=== Opportunity Forecast Results ===")
    print(f"Model MSE: {opportunity_forecast['mse']:.4f}")
    print("\nTop 3 Important Features:")
    print(opportunity_forecast['feature_importance'].head(3))
    
    print("\n=== Lead Scoring Model Results ===")
    print(f"Cross-validation scores: {lead_scoring['cv_scores'].mean():.4f} (+/- {lead_scoring['cv_scores'].std() * 2:.4f})")
    
    print("\n=== At-Risk Reps Analysis ===")
    print(f"Number of at-risk reps identified: {len(at_risk_analysis['at_risk_reps'])}")
    
    # Visualize results
    predictor.visualize_results(opportunity_forecast, 'feature_importance')
    predictor.visualize_results(at_risk_analysis, 'performance_decline') 