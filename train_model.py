# ==========================================================
# MovieIQ - Movie Success Prediction Model
# Author : Shubham Samarpit
# Model  : Random Forest Classifier
# ==========================================================

# -----------------------------
# Import Libraries
# -----------------------------
import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Load Dataset
# -----------------------------
print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("Clean Dataset/movies_clean.csv")

print(df.head())

# -----------------------------
# Handle Missing Genres
# -----------------------------
# ~9% of rows have no genre after cleaning (source data had none listed).
# Coding them as their own category keeps the rows instead of dropping them,
# and keeps prediction-time lookups safe for genres the encoder hasn't seen.
missing_genres = df["primary_genre"].isnull().sum()
print(f"\nMissing genre values: {missing_genres} -> filling with 'Unknown'")

df["primary_genre"] = df["primary_genre"].fillna("Unknown")

# -----------------------------
# Encode Genre
# -----------------------------
print("\nEncoding Genre...")

genre_encoder = LabelEncoder()

df["primary_genre"] = genre_encoder.fit_transform(
    df["primary_genre"]
)

# -----------------------------
# Feature Selection
# -----------------------------
print("\nSelecting Features...")

features = [
    "budget",
    "popularity",
    "runtime",
    "vote_average",
    "primary_genre"
]

target = "success"

X = df[features]
y = df[target]

# -----------------------------
# Train-Test Split
# -----------------------------
print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# -----------------------------
# Train Random Forest Model
# -----------------------------
# NOTE on class_weight="balanced":
# ~81% of titles in this dataset are labeled a success. A plain Random
# Forest learns to predict "hit" for everything and still scores ~81%
# accuracy without actually discriminating flops from hits. Balancing the
# class weights and capping tree depth trades a little raw accuracy for a
# model that actually differentiates - see the README for the honest
# read on how much signal these five features really carry.
print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# -----------------------------
# Make Predictions
# -----------------------------
print("\nMaking Predictions...")

y_pred = model.predict(X_test)

# -----------------------------
# Model Evaluation
# -----------------------------
print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print(f"\nAccuracy : {accuracy:.2%}")
print(f"Macro F1 : {macro_f1:.3f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Feature Importance
# -----------------------------
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

# -----------------------------
# Create Models Folder
# -----------------------------
os.makedirs("Models", exist_ok=True)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(
    model,
    "Models/movie_success_model.pkl"
)

joblib.dump(
    genre_encoder,
    "Models/genre_encoder.pkl"
)

print("\nModel Saved Successfully!")

print("Location : Models/movie_success_model.pkl")

print("Genre Encoder Saved Successfully!")

print("Location : Models/genre_encoder.pkl")

print("\n" + "=" * 60)
print("MovieIQ Model Training Completed Successfully!")
print("=" * 60)