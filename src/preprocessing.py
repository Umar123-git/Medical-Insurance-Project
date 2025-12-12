
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_data(path):
    df = pd.read_csv(path)
    return df

def build_preprocessor(df):
    # Identify columns
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    # exclude target 'charges' if present
    if 'charges' in numeric_cols:
        numeric_cols.remove('charges')
    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    # Build transformers
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    return preprocessor, numeric_cols, categorical_cols

def save_preprocessor(preprocessor, path='models/preprocessor.joblib'):
    joblib.dump(preprocessor, path)
