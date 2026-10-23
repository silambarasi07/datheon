
# Datheon Task 1: Multi-Class Benchmark Classification
# Dataset: iris.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load the actual dataset
df = pd.read_csv("iris.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# 2. Separate features and target
X = df.drop(columns=["species"])
y = df["species"]

# 3. Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Define the two classification models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(random_state=42))
    ])
}

results = []
class_names = sorted(y.unique())

# 5. Train, predict and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print(f"\n{'=' * 45}")
    print(name)
    print(f"{'=' * 45}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

    # 6. Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    plt.tight_layout()
    plt.savefig(
        name.lower().replace(" ", "_") + "_confusion_matrix.png",
        dpi=300
    )
    plt.show()

# 7. Compare model performance
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

# 8. Plot model comparison
results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].plot(kind="bar", figsize=(10, 5), ylim=(0, 1.05))

plt.title("Classification Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()

# 9. Visualize decision boundaries using petal features
# This visualization uses only two features for plotting.

X_2d = df[["petal_length", "petal_width"]]
y_2d = df["species"]

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_2d,
    test_size=0.20,
    random_state=42,
    stratify=y_2d
)

boundary_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(random_state=42))
    ])
}

for name, model in boundary_models.items():
    model.fit(X_train_2d, y_train_2d)

    x_min = X_2d.iloc[:, 0].min() - 0.5
    x_max = X_2d.iloc[:, 0].max() + 0.5
    y_min = X_2d.iloc[:, 1].min() - 0.5
    y_max = X_2d.iloc[:, 1].max() + 0.5

    import numpy as np

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )

    grid = pd.DataFrame({
        "petal_length": xx.ravel(),
        "petal_width": yy.ravel()
    })

    predictions = model.predict(grid)

    # Convert predicted class names to numeric plotting labels
    class_to_number = {
        species: i for i, species in enumerate(class_names)
    }
    Z = np.array([class_to_number[p] for p in predictions])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")

    for species in class_names:
        subset = X_2d[y_2d == species]
        plt.scatter(
            subset["petal_length"],
            subset["petal_width"],
            label=species,
            edgecolors="black",
            s=30
        )

    plt.title(f"Decision Boundary - {name}")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        name.lower().replace(" ", "_") + "_decision_boundary.png",
        dpi=300
    )
    plt.show()

print("\nTask completed successfully!")
