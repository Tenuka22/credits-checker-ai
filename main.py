import pandas as pd
import joblib
import tensorflow as tf

features = {
    "checking_status": "<0",
    "duration": 6,
    "credit_history": "critical/other existing credit",
    "purpose": "radio/tv",
    "credit_amount": 1169,
    "savings_status": "no known savings",
    "employment": ">=7",
    "installment_commitment": 4,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 4,
    "property_magnitude": "real estate",
    "age": 67,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 2,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "yes",
    "foreign_worker": "yes",
}

new_data = pd.DataFrame([features])

model = tf.keras.models.load_model("./models/credits.model.keras")
preprocessor = joblib.load("./models/preprocessor.joblib")

new_data_processed = preprocessor.transform(new_data)
print("Processed data shape:", new_data_processed.shape)
print(new_data_processed)

y_prob = model.predict(new_data_processed)
y_class = (y_prob > 0.5).astype(int)

print("Predicted probability of being 'bad':", y_prob[0][0])
print("Predicted class (0=good, 1=bad):", y_class[0][0])
