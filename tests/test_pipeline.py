import os
import joblib
import pandas as pd

def test_data_file_exists():
    """Verify that cleaned dataset exists, is non-empty, and has expected column count."""
    data_path = "data/cleaned_heart.csv"
    assert os.path.exists(data_path), f"Cleaned dataset missing at {data_path}"
    df = pd.read_csv(data_path)
    assert not df.empty, "Dataset file is empty"
    assert df.shape[1] == 14, f"Expected 14 columns, found {df.shape[1]}"

def test_model_and_scaler_artifacts():
    """Verify that model.pkl and scaler.pkl exist and load successfully."""
    assert os.path.exists("model.pkl"), "model.pkl artifact is missing"
    assert os.path.exists("scaler.pkl"), "scaler.pkl artifact is missing"
    
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    assert model is not None, "Failed to load model"
    assert scaler is not None, "Failed to load scaler"

def test_model_prediction_shape():
    """Test model prediction pipeline on dummy input data."""
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
    
    # Standard sample matching dataset features (13 features)
    dummy_input = [[63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0]]
    scaled_input = scaler.transform(dummy_input)
    prediction = model.predict(scaled_input)
    
    assert len(prediction) == 1, "Model output shape mismatch"
    assert prediction[0] in [0, 1], "Prediction target must be binary (0 or 1)"