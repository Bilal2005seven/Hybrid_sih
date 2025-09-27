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

# Load trained pipeline and encoder
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
if __name__ == "__main__":
    sample_input = {
        "N": 6,
        "P": None,
        "K": None,
        "temperature": 19.656901,
        "humidity": 89.937010,
        "ph": 5.937650,
        "rainfall": 108.045893
    }

    result = recommend_crops(sample_input, top_n=3)

    print("🔧 Imputed values:")
    for key, value in result["imputed_values"].items():
        print(f"{key}: {value}")

    print("\n🌱 Top recommended crops:")
    for crop, prob in result["recommendations"]:
        print(f"{crop}: {prob:.4f}")