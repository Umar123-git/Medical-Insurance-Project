
# Medical Insurance Cost Prediction (Semester Project)

**Dataset:** Medical Insurance Cost Prediction (Kaggle)

Link: https://www.kaggle.com/datasets/rahulvyasm/medical-insurance-cost-prediction



## Contents
- `data/` : sample CSV provided so you can run the app immediately. The real dataset can be downloaded via the included script.
- `src/` : preprocessing and training scripts
- `app/` : Streamlit app that shows EDA, trains model, and allows runtime predictions
- `download_dataset.sh` : script (uses Kaggle CLI) to download the dataset automatically
- `requirements.txt` : Python dependencies
- `medical_insurance_project.zip` : this zip file (created for submission)

## How to use
1. (Optional) Install kaggle CLI and set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`, then run:
   ```bash
   bash download_dataset.sh
   ```
   This will place the dataset CSV into `data/insurance.csv`.

2. Create a Python virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\\Scripts\\activate`
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

4. The app will perform EDA, train a model (or load a saved model), and provide a runtime prediction form.

## Notes
- The project includes a small sample dataset `data/sample_insurance.csv` so you can test everything without downloading Kaggle dataset.
- For best results, download the full Kaggle dataset as described above.

Good luck with your semester project!
