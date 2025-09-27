# agent.py
import json
import pandas as pd
import numpy as np
import joblib

# Create a simple mock class to replace the custom imputer
class KNNImputerWithStore:
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
    def fit_transform(self, X, y=None):
        return X

# Load the pipeline and encoder
pipeline = joblib.load("crop_recommender.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def recommend_crops(input_dict, top_n=3):
    """
    Recommend top-N crops given input features.
    Also returns imputed values of features (like P, K).
    """
    # Convert dict to DataFrame (ensure same column order as training)
    input_df = pd.DataFrame([input_dict], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    
    # Predict probabilities (this also triggers imputation + scaling internally)
    predicted_probabilities = pipeline.predict_proba(input_df)[0]
    
    # For now, just use the input values as imputed values
    imputed_values = input_dict.copy()
    
    # Sort and get top-N indices
    sorted_indices = np.argsort(predicted_probabilities)[::-1][:top_n]
    
    # Map indices back to crop names
    top_crops = label_encoder.inverse_transform(sorted_indices)
    
    # Prepare results
    crop_recommendations = [(crop, float(predicted_probabilities[idx])) for crop, idx in zip(top_crops, sorted_indices)]
    
    return {
        "imputed_values": imputed_values,
        "recommendations": crop_recommendations
    }

from llm_recommend import llm_recommend

def merge_recommendations(ml_output: dict, llm_output: dict, top_n=3):
    f"""
    Merge ML + LLM results and return the best/common top crops.
    Priority:
      1. Common crops in both
      2. Fill remaining slots with highest ML probability crops
      3. convert the overall "explanation" in hindi
    """
    ml_crops = [crop for crop, _ in ml_output["recommendations"]]
    
    # Handle different LLM output structures
    if "recommendations" in llm_output:
        llm_crops = [c["crop_name"] for c in llm_output.get("recommendations", [])]
    elif "top_crops" in llm_output:
        llm_crops = [c["crop"] for c in llm_output.get("top_crops", [])]
    else:
        llm_crops = []

    # Common crops first
    common = [c for c in ml_crops if c in llm_crops]

    # Add ML-only crops if needed
    final = common[:]
    for crop in ml_crops:
        if crop not in final:
            final.append(crop)
        if len(final) >= top_n:
            break

    return {
        "common_crops": common,
        "final_top_crops": final,
        "ml_recommendations": ml_crops,
        "llm_recommendations": llm_crops,
        "explanation": llm_output.get("summary", llm_output.get("explanation", ""))
    }


if __name__ == "__main__":
    # Example input
    sample_input = {
        "N": 6,
        "P": 20,
        "K": 25,
        "temperature": 19.65,
        "humidity": 89.93,
        "ph": 5.93,
        "rainfall": 108.04
    }

    print("🔹 Running ML pipeline...")
    ml_result = recommend_crops(sample_input)

    print("🔹 Running LLM recommendation...")
    llm_result = llm_recommend(sample_input)

    print("\n🔹 Final merged results:")
    merged = merge_recommendations(ml_result, llm_result)
    print(json.dumps(merged, indent=2))
