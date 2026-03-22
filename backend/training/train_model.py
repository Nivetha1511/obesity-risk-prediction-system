from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf


FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "family_history_with_overweight",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS",
]

TARGET_COLUMN = "NObeyesdad"

RISK_LEVEL_MAP = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III",
}


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "dataset" / "obesity_dataset.csv"
    model_dir = project_root / "backend" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)

    missing_columns = [
        col for col in (FEATURE_COLUMNS + [TARGET_COLUMN]) if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_classes = len(np.unique(y))
    model = build_model(input_dim=X_train_scaled.shape[1], num_classes=num_classes)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training epochs used: {len(history.history['loss'])}")

    model.save(model_dir / "ann_obesity_model.keras")

    target_encoder = LabelEncoder()
    target_encoder.fit(y)

    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(target_encoder, model_dir / "target_encoder.pkl")
    joblib.dump(FEATURE_COLUMNS, model_dir / "feature_columns.pkl")
    joblib.dump(RISK_LEVEL_MAP, model_dir / "risk_level_map.pkl")

    print("Model and preprocessing artifacts saved to backend/models/")


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
