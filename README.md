# Fruit Quality Detector AI :apple:

Streamlit app for classifying fruit freshness using a CNN (MobileNetV2) and a Random Forest baseline.

## Features :sparkles:
- Two model options: CNN and Random Forest
- Image upload UI with prediction confidence
- Automatic training when the default model is missing

## Dataset :open_file_folder:
Download: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification/data

Expected structure (only dataset/train is used for training):
```
dataset/
  train/
    freshapples/
    freshbanana/
    freshoranges/
    rottenapples/
    rottenbanana/
    rottenoranges/
```

## Setup :gear:

Windows (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training :brain:
```
python model.py
```
This produces local model files (not tracked in git):
- fruit_classifier_model_*.keras
- fruit_classifier_ml.joblib
- class_names.joblib

## Run the app :rocket:
```
streamlit run app.py
```
If `fruit_classifier_model.keras` is missing, the app will train automatically.

## Notes :memo:
- Dataset and model files are excluded via .gitignore.
- If you use a different dataset, keep the same class folder names or update the code.
