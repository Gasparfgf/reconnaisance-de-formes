import os
import joblib
import numpy as np

from main import extract_features


def predict_image(image_path, model_path="modele_feuilles_rf.pkl", scaler_path="scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    features = extract_features(image_path)
    if features is None:
        print("Aucune forme détectée dans l'image.")
        return

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]

    print(f"Cette feuille est probablement : {prediction.upper()}")


test_image = "./1568.jpg"
if os.path.exists(test_image):
    predict_image(test_image)
else:
    print("Aucune image de test trouvée. Place une image dans './1568.jpg'.")
