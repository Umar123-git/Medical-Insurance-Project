
#!/bin/bash
# Download Kaggle dataset. Requires kaggle CLI configured with environment variables:
# export KAGGLE_USERNAME=your_username
# export KAGGLE_KEY=your_key
# Or run `kaggle config` to set them.
if ! command -v kaggle &> /dev/null
then
    echo "kaggle CLI not found. Install it with: pip install kaggle"
    exit 1
fi
mkdir -p data
kaggle datasets download -d rahulvyasm/medical-insurance-cost-prediction -f insurance.csv -p data --unzip
echo "Dataset downloaded to data/insurance.csv"
