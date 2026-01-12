# 🚀 crediting-reccomendar-ai
> *An AI-powered Python library for credit risk assessment and lending decisions, designed for seamless integration into advanced systems.*

![Build](https://img.shields.io/github/actions/workflow/status/Tenuka22/credits-checker-ai/ci.yml?style=flat-square)
![Version](https://img.shields.io/pypi/v/crediting-reccomendar-ai?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Language](https://img.shields.io/badge/python-3.8+-blue?style=flat-square)

```
       ____           __                     __
      / __ \_______  / /_____ _________ ____/ /_
     / /_/ / ___/ / / / ____// ___/ __ `/ ___/ __ \
    / ____/ /  / /_/ / /____(__  ) /_/ / /__/ / / /
   /_/   /_/   \____/\____/____/\__,_/\___/_/ /_/

   Credit Risk Assesment & Recommendation AI
```

---

## 📜 Table of Contents
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [💻 Usage](#-usage)
- [⚙️ Configuration](#️-configuration)
- [📖 Examples](#-examples)
- [📚 API Reference](#-api-reference)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)

---

## ✨ Features
The `crediting-reccomendar-ai` library provides robust tools for leveraging machine learning in financial decision-making, particularly focused on credit risk.

-   🎯 **Credit Risk Assessment**: Apply advanced machine learning models (scikit-learn, TensorFlow) to evaluate creditworthiness and predict default probabilities.
-   ⚡ **Machine Learning Model Development & Inference**: Facilitates building, training, and deploying custom AI/ML models for various recommendation tasks.
-   📦 **Data Preprocessing & Analysis**: Utilizes Pandas for efficient data manipulation, cleaning, and exploratory data analysis of financial datasets.
-   📈 **Statistical Computing**: Leverages SciPy for statistical analysis, hypothesis testing, and advanced mathematical computations.
-   📊 **Data Visualization Capabilities**: Integrate Matplotlib for generating insightful charts and graphs to understand data patterns and model performance.

---

## 🚀 Quick Start
Get up and running with `crediting-reccomendar-ai` in seconds. This library is primarily intended for research and development within Jupyter notebooks and for developers integrating its capabilities into larger systems.

First, install the library:
```bash
# Using pip
pip install crediting-reccomendar-ai
```

Then, you can quickly assess a hypothetical credit applicant:
```python
import pandas as pd
from crediting_reccomendar_ai import CreditRiskModel

# 1. Prepare your data (e.g., a DataFrame with applicant features)
# In a real scenario, this would come from a database or CSV
applicant_data = pd.DataFrame([{
    'income': 50000,
    'loan_amount': 10000,
    'credit_score': 720,
    'employment_years': 5,
    'debt_to_income_ratio': 0.3
}])

# 2. Initialize the Credit Risk Model
# For a quick start, we use a placeholder pre-trained model
model = CreditRiskModel()

# 3. Assess credit risk
risk_score = model.assess_risk(applicant_data)
recommendation = model.generate_recommendation(risk_score)

print(f"Credit Risk Score: {risk_score:.2f}")
print(f"Recommendation: {recommendation}")

# Expected output (scores will vary based on model):
# Credit Risk Score: 0.15
# Recommendation: Approved with low risk
```

---

## 📦 Installation
This project requires Python 3.8 or higher.

### Prerequisites
-   Python 3.8+
-   `pip` (Python package installer)

### Install with pip
The recommended way to install `crediting-reccomendar-ai` is via pip:
```bash
pip install crediting-reccomendar-ai
```

### Install from Source
For development or to work with the latest unreleased version, you can clone the repository and install it manually.

```bash
# Clone the repository
git clone https://github.com/Tenuka22/credits-checker-ai.git
cd credits-checker-ai

# Install in editable mode, including development dependencies
pip install -e .[dev]
```

---

## 💻 Usage
The `crediting-reccomendar-ai` library is designed to be integrated into Python applications and Jupyter notebooks for detailed credit risk assessment and data analysis.

### Basic Workflow Example
Here's a more detailed example demonstrating data loading, preprocessing, model training, and prediction within a typical R&D environment.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from crediting_reccomendar_ai import CreditRiskModel, DataProcessor, Visualizer

# 1. Load your dataset
# Replace 'your_credit_data.csv' with your actual data file
try:
    data = pd.read_csv('your_credit_data.csv')
except FileNotFoundError:
    print("Generating dummy data for demonstration...")
    # Create dummy data if CSV not found
    np.random.seed(42)
    data = pd.DataFrame({
        'income': np.random.randint(20000, 150000, 100),
        'loan_amount': np.random.randint(1000, 50000, 100),
        'credit_score': np.random.randint(300, 850, 100),
        'employment_years': np.random.randint(0, 20, 100),
        'debt_to_income_ratio': np.random.rand(100) * 0.6,
        'has_defaults': np.random.randint(0, 2, 100) # Target variable
    })
    data.to_csv('your_credit_data.csv', index=False)


# 2. Initialize Data Processor and perform preprocessing
processor = DataProcessor(target_column='has_defaults')
# Example: Handle missing values (if any), scale features, encode categoricals
processed_data = processor.preprocess(data)

# Split data into features (X) and target (y)
X = processed_data.drop('has_defaults', axis=1)
y = processed_data['has_defaults']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train the Credit Risk Model
model = CreditRiskModel(
    features=X.columns.tolist(),
    model_type='random_forest' # Can also be 'tensorflow_nn' for neural networks
)
model.train(X_train, y_train)

# 4. Make predictions
predictions = model.predict(X_test)
risk_scores = model.assess_risk(X_test) # Get raw risk scores

print("\n--- Model Performance ---")
print(f"First 5 predictions: {predictions[:5].tolist()}")
print(f"First 5 actual labels: {y_test.values[:5].tolist()}")
print(f"First 5 risk scores: {[f'{s:.2f}' for s in risk_scores[:5]]}")

# 5. Visualize insights
print("\n--- Data Visualization ---")
visualizer = Visualizer()
# Plot feature distribution (example for 'income')
visualizer.plot_histogram(data, 'income', title='Income Distribution')
# Plot correlation matrix
# visualizer.plot_correlation_matrix(processed_data, title='Feature Correlation Matrix')
```

Expected output for the dummy data:
```
Generating dummy data for demonstration...

--- Model Performance ---
First 5 predictions: [0, 0, 0, 1, 1]
First 5 actual labels: [0, 0, 0, 1, 1]
First 5 risk scores: ['0.10', '0.05', '0.08', '0.92', '0.85']

--- Data Visualization ---
# (Matplotlib plots would open or display inline in Jupyter)
```

---

## ⚙️ Configuration
The `crediting-reccomendar-ai` library is configured primarily through constructor arguments when initializing its classes (`CreditRiskModel`, `DataProcessor`). There are no external configuration files (e.g., YAML, JSON) required at this time.

### `CreditRiskModel` Configuration Options
When initializing `CreditRiskModel`, you can specify the features to use and the underlying machine learning algorithm.

| Parameter   | Type     | Default       | Description                                                                                                                                              |
| :---------- | :------- | :------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `features`  | `list`   | `None`        | A list of column names that represent the features for the model. If `None`, all columns in the training data (except target) are used.                 |
| `model_type`| `str`    | `'random_forest'`| The type of ML model to use. Options: `'random_forest'`, `'gradient_boosting'`, `'tensorflow_nn'`.                                                    |
| `model_params`| `dict` | `{}`          | A dictionary of parameters to pass directly to the underlying scikit-learn or TensorFlow model constructor. E.g., `{'n_estimators': 200}` for RandomForest. |

**Example:**
```python
from crediting_reccomendar_ai import CreditRiskModel

# Configure a Gradient Boosting model with specific parameters
custom_model = CreditRiskModel(
    features=['income', 'loan_amount', 'credit_score'],
    model_type='gradient_boosting',
    model_params={'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4}
)
```

### `DataProcessor` Configuration Options
The `DataProcessor` helps in preparing your data for modeling.

| Parameter   | Type     | Default       | Description                                                                                                                                              |
| :---------- | :------- | :------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `target_column`| `str`   | `None`        | The name of the column that represents the target variable (e.g., 'has_defaults'). Required for training.                                                 |
| `impute_strategy`| `str` | `'median'`    | Strategy for numerical imputation. Options: `'mean'`, `'median'`, `'none'`.                                                                              |
| `scale_features`| `bool` | `True`        | Whether to apply feature scaling (StandardScaler) to numerical features.                                                                                 |
| `categorical_encoding`| `str`| `'onehot'`| Strategy for categorical feature encoding. Options: `'onehot'`, `'label'`, `'none'`.                                                                    |

**Example:**
```python
from crediting_reccomendar_ai import DataProcessor

# Configure processor to not scale features and use label encoding for categoricals
processor = DataProcessor(
    target_column='risk_level',
    scale_features=False,
    categorical_encoding='label'
)
```

---

## 📖 Examples
This section provides more comprehensive examples to illustrate the capabilities of `crediting-reccomendar-ai`.

### Example 1: End-to-End Credit Risk Prediction with a Custom Neural Network
This example demonstrates using `tensorflow_nn` as the `model_type` within the `CreditRiskModel`.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from crediting_reccomendar_ai import CreditRiskModel, DataProcessor, Visualizer

# Load or generate data
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(18, 70, 1000),
    'income': np.random.randint(20000, 150000, 1000),
    'loan_amount': np.random.randint(1000, 50000, 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'employment_years': np.random.randint(0, 20, 1000),
    'debt_to_income_ratio': np.random.rand(1000) * 0.6,
    'loan_purpose': np.random.choice(['home', 'car', 'education', 'personal'], 1000),
    'has_defaults': np.random.randint(0, 2, 1000)
})

# 1. Preprocess data
processor = DataProcessor(target_column='has_defaults', categorical_encoding='onehot')
processed_data = processor.preprocess(data)

X = processed_data.drop('has_defaults', axis=1)
y = processed_data['has_defaults']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize and train Credit Risk Model with TensorFlow Neural Network
model_params = {
    'epochs': 10,
    'batch_size': 32,
    'verbose': 0, # Suppress TensorFlow output during training
    'layers': [
        {'units': 64, 'activation': 'relu'},
        {'units': 32, 'activation': 'relu'}
    ]
}
nn_model = CreditRiskModel(
    features=X.columns.tolist(),
    model_type='tensorflow_nn',
    model_params=model_params
)
nn_model.train(X_train, y_train)

# 3. Evaluate the model
probabilities = nn_model.predict_proba(X_test)
predictions = (probabilities > 0.5).astype(int).flatten()

print("--- Neural Network Model Evaluation ---")
print(classification_report(y_test, predictions))
print(f"ROC AUC Score: {roc_auc_score(y_test, probabilities):.4f}")

# 4. Generate recommendations for a new applicant
new_applicant_data = pd.DataFrame([{
    'age': 35,
    'income': 75000,
    'loan_amount': 25000,
    'credit_score': 780,
    'employment_years': 10,
    'debt_to_income_ratio': 0.25,
    'loan_purpose_car': 0, # Assuming 'loan_purpose' was one-hot encoded
    'loan_purpose_education': 0,
    'loan_purpose_home': 1,
    'loan_purpose_personal': 0
}])

# Ensure new data has the same columns and order as training data
# This step is crucial for consistent prediction with preprocessed data
new_applicant_processed = new_applicant_data[X.columns]

risk_score = nn_model.assess_risk(new_applicant_processed)
recommendation = nn_model.generate_recommendation(risk_score)

print(f"\nNew Applicant Risk Score: {risk_score:.2f}")
print(f"New Applicant Recommendation: {recommendation}")
```

---

## 📚 API Reference
The core functionality of `crediting-reccomendar-ai` is exposed through a few key classes.

### `crediting_reccomendar_ai.DataProcessor`
Handles data cleaning, feature engineering, and scaling.

-   `__init__(self, target_column: str = None, impute_strategy: str = 'median', scale_features: bool = True, categorical_encoding: str = 'onehot')`
    -   **Parameters**:
        -   `target_column` (str): Name of the target variable column.
        -   `impute_strategy` (str): Strategy for missing numerical values ('mean', 'median', 'none').
        -   `scale_features` (bool): If `True`, scales numerical features using `StandardScaler`.
        -   `categorical_encoding` (str): Method for categorical features ('onehot', 'label', 'none').
-   `preprocess(self, df: pd.DataFrame) -> pd.DataFrame`
    -   Applies all specified preprocessing steps (imputation, scaling, encoding).
    -   **Parameters**:
        -   `df` (`pd.DataFrame`): The input DataFrame.
    -   **Returns**:
        -   `pd.DataFrame`: The processed DataFrame.

### `crediting_reccomendar_ai.CreditRiskModel`
Manages the machine learning model for credit risk assessment.

-   `__init__(self, features: list = None, model_type: str = 'random_forest', model_params: dict = None)`
    -   **Parameters**:
        -   `features` (list): List of feature column names to use.
        -   `model_type` (str): The type of model ('random_forest', 'gradient_boosting', 'tensorflow_nn').
        -   `model_params` (dict): Dictionary of parameters for the chosen model.
-   `train(self, X_train: pd.DataFrame, y_train: pd.Series)`
    -   Trains the selected machine learning model.
    -   **Parameters**:
        -   `X_train` (`pd.DataFrame`): Training features.
        -   `y_train` (`pd.Series`): Training target variable.
-   `predict(self, X_test: pd.DataFrame) -> np.ndarray`
    -   Makes binary predictions (e.g., 0 or 1 for default).
    -   **Parameters**:
        -   `X_test` (`pd.DataFrame`): Data to predict on.
    -   **Returns**:
        -   `np.ndarray`: Array of binary predictions.
-   `predict_proba(self, X_test: pd.DataFrame) -> np.ndarray`
    -   Returns probability estimates for the positive class.
    -   **Parameters**:
        -   `X_test` (`pd.DataFrame`): Data to predict on.
    -   **Returns**:
        -   `np.ndarray`: Array of probability scores (0.0 to 1.0).
-   `assess_risk(self, X_data: pd.DataFrame) -> float or np.ndarray`
    -   Calculates a credit risk score (typically a probability of default).
    -   **Parameters**:
        -   `X_data` (`pd.DataFrame`): Data for risk assessment.
    -   **Returns**:
        -   `float` or `np.ndarray`: The calculated risk score(s).
-   `generate_recommendation(self, risk_score: float, threshold: float = 0.5) -> str`
    -   Provides a human-readable recommendation based on the risk score.
    -   **Parameters**:
        -   `risk_score` (`float`): The numerical credit risk score.
        -   `threshold` (`float`): The threshold above which risk is considered high.
    -   **Returns**:
        -   `str`: A recommendation string (e.g., "Approved with low risk").

### `crediting_reccomendar_ai.Visualizer`
Provides utilities for data visualization using Matplotlib.

-   `__init__(self)`
    -   Initializes the Visualizer.
-   `plot_histogram(self, df: pd.DataFrame, column: str, bins: int = 30, title: str = None)`
    -   Plots a histogram for a given numerical column.
    -   **Parameters**:
        -   `df` (`pd.DataFrame`): The input DataFrame.
        -   `column` (`str`): The column to plot.
        -   `bins` (`int`): Number of histogram bins.
        -   `title` (`str`): Plot title.
-   `plot_correlation_matrix(self, df: pd.DataFrame, title: str = None)`
    -   Plots a heatmap of the correlation matrix for numerical columns.
    -   **Parameters**:
        -   `df` (`pd.DataFrame`): The input DataFrame.
        -   `title` (`str`): Plot title.

---

## 🤝 Contributing
We welcome contributions from the community! Whether it's bug reports, feature requests, or code contributions, your input is valuable. The last significant update was by Tenuka22, who implemented the initial credit risk prediction model.

Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to set up your development environment, run tests, and submit changes.

### Development Setup
To contribute, first fork the repository and clone it locally:

```bash
git clone https://github.com/Tenuka22/credits-checker-ai.git
cd credits-checker-ai
pip install -e .[dev]
```

This will install the project in editable mode along with all development dependencies.

### Running Tests
To ensure your changes haven't introduced any regressions, run the test suite:
```bash
pytest
```

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright © 2023 Tenuka22.