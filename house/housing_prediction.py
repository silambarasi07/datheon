# Real Estate Housing Price Prediction
# Task 2: Linear Regression, Ridge Regression, and Lasso Regression
# Dataset: housing.csv (California Housing dataset)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
DATA_FILE = "housing.csv"
TARGET = "median_house_value"

# 1. Load the dataset
df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found in housing.csv.")

# 2. Basic cleaning and feature engineering
# Remove duplicate rows, if any.
df = df.drop_duplicates().copy()

# Add useful features using existing columns.
df["rooms_per_household"] = df["total_rooms"] / df["households"].replace(0, np.nan)
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"].replace(0, np.nan)
df["population_per_household"] = df["population"] / df["households"].replace(0, np.nan)

# Add an interaction feature between income and rooms.
df["income_x_rooms"] = df["median_income"] * df["total_rooms"]

# Remove rows with missing target only; predictor missing values are imputed later.
df = df.dropna(subset=[TARGET])

# 3. Explore numeric correlations
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df[TARGET], bins=40, kde=True)
plt.title("Distribution of Median House Value")
plt.xlabel("Median house value")
plt.tight_layout()
plt.show()

# 4. Separate input features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# Split before fitting preprocessors to avoid data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

# 5. Preprocessing
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 6. Define models and tune Ridge/Lasso with GridSearchCV
ols_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

ridge_model = GridSearchCV(
    estimator=Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge())
    ]),
    param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1
)

lasso_model = GridSearchCV(
    estimator=Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", Lasso(max_iter=50000, tol=0.001))
    ]),
    param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0]},
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1
)

models = {
    "Linear Regression (OLS)": ols_model,
    "Ridge Regression": ridge_model,
    "Lasso Regression": lasso_model
}

# 7. Train, predict, and evaluate
results = []
predictions = {}
fitted_models = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    fitted_models[model_name] = model

    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    results.append({
        "Model": model_name,
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2 Score": r2_score(y_test, y_pred)
    })

    if isinstance(model, GridSearchCV):
        print("Best parameters:", model.best_params_)
        print("Best cross-validation RMSE:", round(-model.best_score_, 3))

results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

# Save metric table for project submission.
results_df.to_csv("model_comparison_results.csv", index=False)
print("\nSaved model metrics to model_comparison_results.csv")

# 8. Plot actual versus predicted values
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (model_name, y_pred) in zip(axes, predictions.items()):
    ax.scatter(y_test, y_pred, alpha=0.25, s=10)
    lower = min(y_test.min(), np.min(y_pred))
    upper = max(y_test.max(), np.max(y_pred))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="red")
    ax.set_title(model_name)
    ax.set_xlabel("Actual house value")
    ax.set_ylabel("Predicted house value")
plt.tight_layout()
plt.show()

# 9. Residual plots and distributions
fig, axes = plt.subplots(3, 2, figsize=(13, 13))
for row, (model_name, y_pred) in enumerate(predictions.items()):
    residuals = y_test.to_numpy() - y_pred

    axes[row, 0].scatter(y_pred, residuals, alpha=0.25, s=10)
    axes[row, 0].axhline(0, linestyle="--", color="red")
    axes[row, 0].set_title(f"{model_name}: Residuals vs Predicted")
    axes[row, 0].set_xlabel("Predicted value")
    axes[row, 0].set_ylabel("Residual")

    sns.histplot(residuals, bins=40, kde=True, ax=axes[row, 1])
    axes[row, 1].set_title(f"{model_name}: Residual Distribution")
    axes[row, 1].set_xlabel("Actual - predicted")
plt.tight_layout()
plt.show()

# 10. Print the best model based on test RMSE
best_model = results_df.iloc[0]
print("\nFINAL CONCLUSION")
print(f"Best model by test RMSE: {best_model['Model']}")
print(f"RMSE: {best_model['RMSE']:.4f}")
print(f"MAE: {best_model['MAE']:.4f}")
print(f"R2 Score: {best_model['R2 Score']:.4f}")
print(
    "\nThe model with the lowest test RMSE had the smallest root mean squared "
    "prediction error among the three evaluated models. Consider RMSE, MAE, "
    "R2, and the residual plots together when discussing performance."
)
