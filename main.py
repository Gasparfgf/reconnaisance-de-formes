import os
import cv2
import numpy as np
import pandas as pd


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-5)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h != 0 else 0
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    hu_moments = cv2.HuMoments(cv2.moments(cnt)).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    features = [
        area, perimeter, aspect_ratio, extent,
        solidity, compactness
    ] + list(hu_moments)

    return features


def build_dataset(dataset_path="dataset"):
    classes = ["chene", "erable", "bouleau", "saule"]
    data, labels = [], []

    for cls in classes:
        folder = os.path.join(dataset_path, cls)
        if not os.path.isdir(folder):
            print(f"Dossier introuvable : {folder}")
            continue

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            features = extract_features(path)
            if features is not None:
                data.append(features)
                labels.append(cls)

    columns = [
        "area", "perimeter", "aspect_ratio", "extent",
        "solidity", "compactness",
        "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"
    ]
    df = pd.DataFrame(data, columns=columns)
    df["label"] = labels

    print(df.head())
    print("\nNombre total d'images traitées :", len(df))

    df.to_csv("feuilles_features.csv", index=False)
    print("Dataset sauvegardé dans feuilles_features.csv")

    return df

