import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dagshub
dagshub.init(repo_owner='Harshhuu1', repo_name='MLFlow_', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Harshhuu1/MLFlow_.mlflow")
mlflow.set_experiment("auto_experminet")

# Set experiment (important)
mlflow.autolog()


# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define params
max_depth = 10
n_estimators = 10
mlflow.set_experiment("01_Random_Forest_Wine_Classification")
with mlflow.start_run():

    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Log metrics + params
    

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save image locally
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
   
    mlflow.log_artifact(__file__)
mlflow.set_tags({
    "Author": "Harsh Yadav",
    "Version": "1.0"
})
