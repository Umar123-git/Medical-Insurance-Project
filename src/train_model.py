
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np

# Handle both relative imports (when called as module) and direct execution
try:
    from .preprocessing import build_preprocessor, load_data
except ImportError:
    from preprocessing import build_preprocessor, load_data

# Get the project root directory (parent of src directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train(path='data/insurance.csv', use_sample=False):
    if use_sample:
        path = 'data/sample_insurance.csv'
    # Construct full path relative to project root
    full_path = os.path.join(PROJECT_ROOT, path)
    df = load_data(full_path)
    # Simple cleaning
    df = df.dropna()
    X = df.drop(columns=['charges'])
    y = df['charges']
    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    # Try two models and pick best
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Pipeline for models
    from sklearn.pipeline import Pipeline
    models = {
        'lr': Pipeline([('pre', preprocessor), ('model', LinearRegression())]),
        'rf': Pipeline([('pre', preprocessor), ('model', RandomForestRegressor(n_jobs=-1, random_state=42))])
    }
    best = None
    best_score = float('inf')
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        print(f"Model {name} RMSE: {rmse}")
        if rmse < best_score:
            best_score = rmse
            best = pipe
    # Save best model
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'best_model.joblib')
    joblib.dump(best, model_path)
    print(f'Saved best model to {model_path}')
    return best

if __name__=='__main__':
    import os
    os.makedirs('models', exist_ok=True)
    train(use_sample=True)
