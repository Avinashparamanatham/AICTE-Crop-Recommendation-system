# Crop Recommendation System - Model Training Script
# Updated for Streamlit deployment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_data(filepath='Crop_recommendation.csv'):
    """Load crop recommendation dataset"""
    try:
        print("Loading dataset...")
        data = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Crops: {data['label'].unique()}")
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found!")
        print("Creating sample dataset for demonstration...")
        return create_sample_dataset()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def create_sample_dataset():
    """Create a sample dataset if the original is not available"""
    print("Generating sample crop recommendation dataset...")
    
    np.random.seed(42)
    crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'jute', 'coffee', 'apple', 'banana', 'grapes']
    
    # Define typical ranges for each crop (simplified)
    crop_profiles = {
        'rice': {'N': (80, 120), 'P': (40, 80), 'K': (40, 80), 'temp': (20, 30), 'humidity': (80, 95), 'ph': (5.5, 7.0), 'rainfall': (150, 300)},
        'wheat': {'N': (50, 100), 'P': (30, 70), 'K': (30, 70), 'temp': (15, 25), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (50, 100)},
        'maize': {'N': (60, 120), 'P': (40, 80), 'K': (60, 100), 'temp': (18, 27), 'humidity': (60, 80), 'ph': (5.8, 7.0), 'rainfall': (50, 150)},
        'cotton': {'N': (100, 150), 'P': (50, 100), 'K': (50, 100), 'temp': (21, 30), 'humidity': (50, 80), 'ph': (5.8, 8.0), 'rainfall': (50, 150)},
        'sugarcane': {'N': (100, 150), 'P': (50, 100), 'K': (100, 150), 'temp': (21, 30), 'humidity': (75, 85), 'ph': (6.0, 7.5), 'rainfall': (75, 150)},
        'jute': {'N': (40, 80), 'P': (20, 60), 'K': (40, 80), 'temp': (24, 35), 'humidity': (70, 80), 'ph': (6.0, 7.5), 'rainfall': (150, 250)},
        'coffee': {'N': (70, 120), 'P': (30, 70), 'K': (60, 120), 'temp': (15, 28), 'humidity': (50, 70), 'ph': (6.0, 7.0), 'rainfall': (150, 250)},
        'apple': {'N': (20, 50), 'P': (125, 200), 'K': (200, 300), 'temp': (15, 25), 'humidity': (50, 60), 'ph': (5.5, 7.0), 'rainfall': (100, 180)},
        'banana': {'N': (50, 100), 'P': (75, 150), 'K': (50, 100), 'temp': (26, 30), 'humidity': (75, 85), 'ph': (6.0, 7.5), 'rainfall': (75, 150)},
        'grapes': {'N': (20, 50), 'P': (125, 200), 'K': (200, 300), 'temp': (15, 25), 'humidity': (45, 65), 'ph': (6.0, 8.0), 'rainfall': (50, 125)}
    }
    
    data = []
    for crop, profile in crop_profiles.items():
        for _ in range(220):  # Generate 220 samples per crop
            sample = {
                'N': np.random.uniform(profile['N'][0], profile['N'][1]),
                'P': np.random.uniform(profile['P'][0], profile['P'][1]),
                'K': np.random.uniform(profile['K'][0], profile['K'][1]),
                'temperature': np.random.uniform(profile['temp'][0], profile['temp'][1]),
                'humidity': np.random.uniform(profile['humidity'][0], profile['humidity'][1]),
                'ph': np.random.uniform(profile['ph'][0], profile['ph'][1]),
                'rainfall': np.random.uniform(profile['rainfall'][0], profile['rainfall'][1]),
                'label': crop
            }
            data.append(sample)
    
    df = pd.DataFrame(data)
    df.to_csv('Crop_recommendation.csv', index=False)
    print(f"Sample dataset created and saved with shape: {df.shape}")
    return df

def perform_eda(data):
    """Perform exploratory data analysis"""
    print("\nPerforming Exploratory Data Analysis...")
    
    # Basic info
    print(f"Dataset shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Unique crops: {data['label'].nunique()}")
    
    # Crop distribution
    print("\nCrop distribution:")
    print(data['label'].value_counts())
    
    try:
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        numerical_data = data.drop('label', axis=1)
        sns.heatmap(numerical_data.corr(), annot=True, cmap='viridis', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Correlation heatmap saved.")
        
        # Feature distributions
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for feature in features:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='label', y=feature, data=data)
            plt.title(f'Distribution of {feature} by Crop')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'visualizations/{feature}_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Feature distribution plots saved.")
        
    except Exception as e:
        print(f"Error in EDA visualization: {str(e)}")

def preprocess_data(data):
    """Preprocess data for model training"""
    print("\nPreprocessing data...")
    
    # Separate features and target
    X = data.drop('label', axis=1)
    y = data['label']
    
    print(f"Features: {list(X.columns)}")
    print(f"Target classes: {y.unique()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train):
    """Train and compare multiple models"""
    print("\nTraining models...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        results[name] = {
            'cv_mean': mean_cv_score,
            'cv_std': std_cv_score,
            'model': model
        }
        
        print(f"{name} CV Score: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
        
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with CV score: {best_score:.4f}")
    
    # Hyperparameter tuning for the best model
    if best_name == 'Random Forest':
        print("Performing hyperparameter tuning for Random Forest...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Fit the best model
    best_model.fit(X_train, y_train)
    
    return best_model, results

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    try:
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test, y_pred)
        classes = np.unique(y_test)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Confusion matrix saved.")
    except Exception as e:
        print(f"Error creating confusion matrix: {str(e)}")
    
    return accuracy

def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance"""
    if hasattr(model, 'feature_importances_'):
        print("\nFeature Importance Analysis:")
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print importance
        for i in range(len(importances)):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # Plot importance
        try:
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), 
                      [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Feature importance plot saved.")
        except Exception as e:
            print(f"Error creating feature importance plot: {str(e)}")

def save_model(model, scaler, filename='models/crop_recommendation_model.pkl'):
    """Save the trained model and scaler"""
    print(f"\nSaving model to {filename}...")
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    }
    
    with open(filename, 'wb') as file:
        pickle.dump(model_data, file)
    
    print("Model saved successfully!")

def test_model(model, scaler):
    """Test the model with sample predictions"""
    print("\nTesting model with sample predictions...")
    
    test_cases = [
        {
            'name': 'Rice conditions',
            'values': [90, 60, 60, 25, 85, 6.5, 200],
            'expected': 'rice'
        },
        {
            'name': 'Wheat conditions', 
            'values': [70, 50, 40, 20, 60, 7.0, 75],
            'expected': 'wheat'
        },
        {
            'name': 'Cotton conditions',
            'values': [120, 75, 75, 25, 70, 7.5, 100],
            'expected': 'cotton'
        }
    ]
    
    for test_case in test_cases:
        features = np.array([test_case['values']])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            max_prob = np.max(probabilities)
            print(f"{test_case['name']}: Predicted={prediction}, Confidence={max_prob:.3f}")
        else:
            print(f"{test_case['name']}: Predicted={prediction}")

def main():
    """Main execution pipeline"""
    print("=== Crop Recommendation System - Model Training ===\n")
    
    try:
        # Step 1: Load data
        data = load_data()
        if data is None:
            return
        
        # Step 2: EDA
        perform_eda(data)
        
        # Step 3: Preprocess
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
        
        # Step 4: Train models
        best_model, results = train_models(X_train, y_train)
        
        # Step 5: Evaluate
        accuracy = evaluate_model(best_model, X_test, y_test)
        
        # Step 6: Feature importance
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        analyze_feature_importance(best_model, feature_names)
        
        # Step 7: Save model
        save_model(best_model, scaler)
        
        # Step 8: Test model
        test_model(best_model, scaler)
        
        print(f"\n=== Training Complete ===")
        print(f"Final model accuracy: {accuracy:.4f}")
        print("Model and visualizations saved successfully!")
        print("Ready for Streamlit deployment!")
        
    except Exception as e:
        print(f"Error in main pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()