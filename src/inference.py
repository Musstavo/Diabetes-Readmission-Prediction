import sys
import os
import pandas as pd
import xgboost as xgb
from preprocess import run_pipeline


def predict(csv_path, model_path="src/diabetes_xgb_model.json"):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    raw_data = pd.read_csv(csv_path)
    processed_data = run_pipeline(raw_data)

    booster = model.get_booster()
    model_features = booster.feature_names

    if model_features is None:
        print("Error: Model does not contain feature names.")
        sys.exit(1)

    for col in model_features:
        if col not in processed_data.columns:
            processed_data[col] = 0

    processed_data = processed_data[model_features]

    results = pd.DataFrame(
        {
            "encounter_id": raw_data.get("encounter_id", range(len(raw_data))),
            "readmission_probability": probs,
        }
    )

    output_file = "predictions.csv"
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py <path_to_patient_data.csv>")
        sys.exit(1)

    predict(sys.argv[1])
