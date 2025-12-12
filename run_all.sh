
#!/bin/bash
# Create models dir, install requirements, train model on sample, then run streamlit
python -c "import sys; print('Python', sys.version)"
python -m pip install -r requirements.txt
mkdir -p models
python -c "from src.train_model import train; train(use_sample=True)"
streamlit run app/streamlit_app.py
