# Brainwave-matrix-intern 
# Fraud Detection Model for Credit Card Transaction 
import numpy as np
import pandas as pd
import matplotlib.pyplot
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_score, recall_score, f1_score,
                           precision_recall_curve, average_precision_score,
                           roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from pyod.models.knn import KNN
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=== 1. Data Loading and Exploration ===")
try:
    df = pd.read_csv('creditcard.csv')
    print("Data loaded successfully")
except FileNotFoundError:
    print("Dataset not found. Creating synthetic data for demonstration...")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=10000, n_features=30, n_informative=15,
                              n_redundant=5, weights=[0.99], flip_y=0, random_state=42)
    df = pd.DataFrame(X)
    df['Class'] = y
    print("Synthetic data created")

print("\nDataset Info:")
print(df.info())
print("\nClass Distribution:")
print(df['Class'].value_counts(normalize=True))
print("\nBasic Statistics:")
print(df.describe())

matplotlib.pyplot.figure(figsize=(10, 6))
sns.countplot(x='Class', data=df)
matplotlib.pyplot.title('Class Distribution (0: Normal, 1: Fraud)')
matplotlib.pyplot.show()

print("\n=== 2. Data Preprocessing ===")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining set shape:", X_train_scaled.shape)
print("Test set shape:", X_test_scaled.shape)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name=""):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    print(f"\n=== {model_name} ===")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    if y_proba is not None:
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"Average Precision Score: {average_precision_score(y_test, y_proba):.4f}")
    if y_proba is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        matplotlib.pyplot.figure(figsize=(8, 6))
        matplotlib.pyplot.plot(recall, precision, marker='.')
        matplotlib.pyplot.xlabel('Recall')
        matplotlib.pyplot.ylabel('Precision')
        matplotlib.pyplot.title(f'Precision-Recall Curve - {model_name}')
        matplotlib.pyplot.show()
    metrics = {
        'Model': model_name,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'PR AUC': average_precision_score(y_test, y_proba) if y_proba is not None else None
    }
    return metrics

print("\n=== 4. Supervised Learning Models ===")
baseline_lr = LogisticRegression(max_iter=1000, random_state=42)
metrics_baseline = evaluate_model(baseline_lr, X_train_scaled, y_train, X_test_scaled, y_test, "Baseline Logistic Regression")

over = SMOTE(sampling_strategy=0.1, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

def create_pipeline(model):
    return Pipeline([
        ('over', over),
        ('under', under),
        ('model', model)
    ])

lr_smote = create_pipeline(LogisticRegression(max_iter=1000, random_state=42))
metrics_lr_smote = evaluate_model(lr_smote, X_train_scaled, y_train, X_test_scaled, y_test, "Logistic Regression with SMOTE")

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
metrics_rf = evaluate_model(rf, X_train_scaled, y_train, X_test_scaled, y_test, "Random Forest")

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',
    random_state=42,
    n_jobs=-1
)
metrics_xgb = evaluate_model(xgb_model, X_train_scaled, y_train, X_test_scaled, y_test, "XGBoost")

lgb_model = lgb.LGBMClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
metrics_lgb = evaluate_model(lgb_model, X_train_scaled, y_train, X_test_scaled, y_test, "LightGBM")

print("\n=== 5. Anomaly Detection Models ===")
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=float(len(y[y == 1])) / float(len(y)),
    random_state=42
)
iso_forest.fit(X_train_scaled)
y_pred_iso = iso_forest.predict(X_test_scaled)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)

print("\n=== Isolation Forest ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_iso))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_iso))

metrics_iso = {
    'Model': 'Isolation Forest',
    'Precision': precision_score(y_test, y_pred_iso),
    'Recall': recall_score(y_test, y_pred_iso),
    'F1': f1_score(y_test, y_pred_iso),
    'ROC AUC': None,
    'PR AUC': None
}

oc_svm = OneClassSVM(
    nu=0.01,
    kernel='rbf',
    gamma='scale'
)
normal_idx = y_train == 0
oc_svm.fit(X_train_scaled[normal_idx])
y_pred_oc = oc_svm.predict(X_test_scaled)
y_pred_oc = np.where(y_pred_oc == -1, 1, 0)

print("\n=== One-Class SVM ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_oc))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_oc))

metrics_oc = {
    'Model': 'One-Class SVM',
    'Precision': precision_score(y_test, y_pred_oc),
    'Recall': recall_score(y_test, y_pred_oc),
    'F1': f1_score(y_test, y_pred_oc),
    'ROC AUC': None,
    'PR AUC': None
}

knn_ad = KNN(contamination=float(len(y[y == 1])) / float(len(y)))
knn_ad.fit(X_train_scaled)
y_scores_knn = knn_ad.decision_function(X_test_scaled)
threshold = np.percentile(y_scores_knn, 100 * (1 - float(len(y[y == 1])) / float(len(y))))
y_pred_knn = np.where(y_scores_knn > threshold, 1, 0)

print("\n=== KNN Anomaly Detection ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

metrics_knn = {
    'Model': 'KNN Anomaly Detection',
    'Precision': precision_score(y_test, y_pred_knn),
    'Recall': recall_score(y_test, y_pred_knn),
    'F1': f1_score(y_test, y_pred_knn),
    'ROC AUC': None,
    'PR AUC': None
}

print("\n=== 6. Model Comparison ===")
results = [
    metrics_baseline,
    metrics_lr_smote,
    metrics_rf,
    metrics_xgb,
    metrics_lgb,
    metrics_iso,
    metrics_oc,
    metrics_knn
]

results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print(results_df.sort_values(by='F1', ascending=False))

print("\n=== 7. Final Model Deployment ===")
final_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',
    random_state=42,
    n_jobs=-1
)

X_scaled = scaler.fit_transform(X)
final_model.fit(X_scaled, y)

joblib.dump(final_model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved to disk")

loaded_model = joblib.load('fraud_detection_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

example_txn = X_test.iloc[0:1]
example_txn_scaled = loaded_scaler.transform(example_txn)
prediction = loaded_model.predict(example_txn_scaled)
probability = loaded_model.predict_proba(example_txn_scaled)[0, 1]

print(f"\nExample Transaction Prediction: {'Fraud' if prediction[0] == 1 else 'Normal'}")
print(f"Fraud Probability: {probability:.4f}")

print("\nFraud detection model implementation complete!")
