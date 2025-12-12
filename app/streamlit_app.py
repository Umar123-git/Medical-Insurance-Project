
import streamlit as st
import pandas as pd
import joblib, os
import sys

# Get the project root directory (parent of app directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to Python path so we can import src module
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_data, build_preprocessor

st.set_page_config(page_title='Medical Insurance Cost Prediction', layout='wide')

st.title('Medical Insurance Cost Prediction')

st.sidebar.header('Options')
use_sample = st.sidebar.checkbox('Use sample dataset (included)', value=True)
if st.sidebar.button('Download Kaggle dataset (requires kaggle CLI)'):
    st.sidebar.write('Run download_dataset.sh from project root to download dataset.')

data_path = os.path.join(PROJECT_ROOT, 'data/sample_insurance.csv') if use_sample else os.path.join(PROJECT_ROOT, 'data/insurance.csv')
if not os.path.exists(data_path):
    st.error(f"Dataset not found at {data_path}. Either enable sample dataset or download Kaggle dataset via download_dataset.sh")
    st.stop()

# Load data
df = pd.read_csv(data_path)

st.header('Introduction')
st.markdown(
    """
    This app explores a medical insurance dataset and provides an interactive
    model to estimate charges based on patient characteristics.
    Use the **EDA** section to understand the data, then try the
    **Runtime Prediction** form below.
    """
)

# High‑level preview
col1, col2 = st.columns((2, 1))
with col1:
    st.subheader('Data records')
    # Show the full dataset in a scrollable table
    st.dataframe(df, height=400, width='stretch')
with col2:
    st.subheader('Dataset overview')
    st.metric('Rows', f"{len(df):,}")
    st.metric('Columns', df.shape[1])
    st.write('Columns by type:')
    type_summary = (
        df.dtypes.reset_index()
        .rename(columns={'index': 'column', 0: 'dtype'})
    )
    # Cast dtype to string to avoid Arrow conversion issues
    type_summary['dtype'] = type_summary['dtype'].astype(str)
    st.table(type_summary)

st.header('Exploratory Data Analysis')

tab_overview, tab_dist, tab_relationships = st.tabs([
    'Summary & Correlations',
    'Distributions',
    'Relationships & Groups',
])

numeric_df = df.select_dtypes(include=['int64', 'float64'])

with tab_overview:
    st.subheader('Summary statistics (numeric features)')
    st.dataframe(numeric_df.describe().T.style.background_gradient(cmap='Blues'))

    st.subheader('Correlation matrix')
    corr = numeric_df.corr()
    st.dataframe(corr.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1))

with tab_dist:
    st.subheader('Distribution of charges')
    # Histogram of charges
    hist_series = df['charges']
    # bin charges and plot counts as a bar chart
    hist_counts = hist_series.value_counts(bins=30, sort=False).sort_index()
    st.bar_chart(hist_counts)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader('Age distribution')
        st.bar_chart(df['age'].value_counts().sort_index())
    with col_b:
        st.subheader('BMI distribution (binned)')
        bmi_binned = pd.cut(df['bmi'], bins=20)
        bmi_counts = bmi_binned.value_counts().sort_index()
        # Use bin midpoints as numeric x-axis for a clean bar chart
        bmi_df = bmi_counts.reset_index()
        bmi_df.columns = ['bmi_bin', 'count']
        bmi_df['bmi_bin_mid'] = bmi_df['bmi_bin'].apply(lambda x: x.mid)
        st.bar_chart(bmi_df.set_index('bmi_bin_mid')['count'])

with tab_relationships:
    st.subheader('Average charges by categorical features')
    col1, col2, col3 = st.columns(3)

    with col1:
        by_smoker = df.groupby('smoker', as_index=False)['charges'].mean()
        st.write('Smoker vs. average charges')
        st.bar_chart(by_smoker.set_index('smoker'))

    with col2:
        by_sex = df.groupby('sex', as_index=False)['charges'].mean()
        st.write('Sex vs. average charges')
        st.bar_chart(by_sex.set_index('sex'))

    with col3:
        by_region = df.groupby('region', as_index=False)['charges'].mean()
        st.write('Region vs. average charges')
        st.bar_chart(by_region.set_index('region'))

    st.subheader('Charges vs. age (line view)')
    age_charges = (
        df.groupby('age', as_index=False)['charges']
        .mean()
        .sort_values('age')
    )
    st.line_chart(age_charges.set_index('age'))

    st.subheader('Scatter: BMI vs charges (colored by smoker)')
    scatter_data = df[['bmi', 'charges', 'smoker']].copy()
    st.scatter_chart(scatter_data, x='bmi', y='charges', color='smoker')

# Allow runtime prediction
st.header('Runtime Prediction')
st.write('Enter values to predict insurance charges:')

with st.form('predict_form'):
    age = st.number_input('Age', min_value=0, max_value=120, value=int(df['age'].median()))
    sex = st.selectbox('Sex', options=sorted(df['sex'].unique()))
    bmi = st.number_input('BMI', min_value=0.0, max_value=80.0, value=float(df['bmi'].median()))
    children = st.number_input('Children', min_value=0, max_value=10, value=int(df['children'].median()))
    smoker = st.selectbox('Smoker', options=sorted(df['smoker'].unique()))
    region = st.selectbox('Region', options=sorted(df['region'].unique()))
    submitted = st.form_submit_button('Predict')
    if submitted:
        # Assemble a single-row dataframe
        sample = pd.DataFrame([{'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}])
        # Load or train model
        model_path = os.path.join(PROJECT_ROOT, 'models/best_model.joblib')
        if not os.path.exists(model_path):
            st.info('No trained model found - training a model on the sample dataset (this may take a moment)...')
            import src.train_model as trainer
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            trainer.train(use_sample=True)
        model = joblib.load(model_path)
        preds = model.predict(sample)
        st.success(f'Estimated insurance charge: {preds[0]:.2f}')
