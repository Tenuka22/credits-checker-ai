import pandas as pd
import joblib
import tensorflow as tf
from typing import TypedDict, Literal


class CreditFeatures(TypedDict):
    checking_status: Literal["<0", "0<=X<200", ">=200", "no checking"]
    duration: float
    credit_history: Literal[
        "no credits/all paid",
        "all paid",
        "existing paid",
        "delayed previously",
        "critical/other existing credit",
    ]
    purpose: Literal[
        "new car",
        "used car",
        "furniture/equipment",
        "radio/tv",
        "domestic appliance",
        "repairs",
        "education",
        "vacation",
        "retraining",
        "business",
        "other",
    ]
    credit_amount: float
    savings_status: Literal[
        "<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"
    ]
    employment: Literal["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]
    installment_commitment: float
    personal_status: Literal[
        "male div/sep",
        "female div/dep/mar",
        "male single",
        "male mar/wid",
        "female single",
    ]
    other_parties: Literal["none", "co applicant", "guarantor"]
    residence_since: float
    property_magnitude: Literal[
        "real estate", "life insurance", "car", "no known property"
    ]
    age: float
    other_payment_plans: Literal["bank", "stores", "none"]
    housing: Literal["rent", "own", "for free"]
    existing_credits: float
    job: Literal[
        "unemp/unskilled non res",
        "unskilled resident",
        "skilled",
        "high qualif/self emp/mgmt",
    ]
    num_dependents: float
    own_telephone: Literal["none", "yes"]
    foreign_worker: Literal["yes", "no"]


def main():
    model = tf.keras.models.load_model("./models/credits.model.keras")
    print("✅ Model loaded")

    preprocessor = joblib.load("./models/preprocessor.joblib")
    print("✅ Preprocessor loaded")

    label_encoder = joblib.load("./models/label_encoder.joblib")
    print("✅ Label encoder loaded")

    feature_order = joblib.load("./models/credits_features.joblib")
    print("✅ Feature order loaded")

    features: CreditFeatures = {
        "checking_status": "<0",
        "duration": 6.0,
        "credit_history": "critical/other existing credit",
        "purpose": "radio/tv",
        "credit_amount": 1169.0,
        "savings_status": "no known savings",
        "employment": ">=7",
        "installment_commitment": 4.0,
        "personal_status": "male single",
        "other_parties": "none",
        "residence_since": 4.0,
        "property_magnitude": "real estate",
        "age": 67.0,
        "other_payment_plans": "none",
        "housing": "own",
        "existing_credits": 2.0,
        "job": "skilled",
        "num_dependents": 1.0,
        "own_telephone": "yes",
        "foreign_worker": "yes",
    }

    new_data = pd.DataFrame([features])

    new_data = new_data[feature_order]

    print("\n📥 Input data:")
    print(new_data)

    new_data_processed = preprocessor.transform(new_data)

    print("\n⚙️ Processed input:")
    print("Shape:", new_data_processed.shape)

    y_prob = model.predict(new_data_processed, verbose=0)
    y_class_index = (y_prob > 0.5).astype(int)
    y_class_label = label_encoder.inverse_transform(y_class_index.ravel())

    print("\n🧠 Prediction Result")
    print(f"Predicted probability of being 'good': {y_prob[0][0]:.4f}")
    print(f"Predicted class: {y_class_label[0]}")


if __name__ == "__main__":
    main()
