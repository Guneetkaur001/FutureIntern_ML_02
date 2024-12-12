import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def create_sample_data(n_samples=10000):
   
    n_normal = int(n_samples * 0.9)
    n_fraud = n_samples - n_normal
  
    normal_features = {
        'amount': np.random.normal(100, 50, n_normal),
        'time': np.random.uniform(0, 24, n_normal),
        'distance_from_home': np.random.normal(10, 5, n_normal),
        'distance_from_last_transaction': np.random.normal(5, 3, n_normal)
    }
  
    fraud_features = {
        'amount': np.random.normal(750, 200, n_fraud),
        'time': np.random.uniform(0, 24, n_fraud),
        'distance_from_home': np.random.normal(50, 20, n_fraud),
        'distance_from_last_transaction': np.random.normal(25, 10, n_fraud)
    }
  
    features = {
        'amount': np.concatenate([normal_features['amount'], fraud_features['amount']]),
        'time': np.concatenate([normal_features['time'], fraud_features['time']]),
        'distance_from_home': np.concatenate([normal_features['distance_from_home'], fraud_features['distance_from_home']]),
        'distance_from_last_transaction': np.concatenate([normal_features['distance_from_last_transaction'], fraud_features['distance_from_last_transaction']])
    }
 
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
 
    df = pd.DataFrame(features)
    df['Class'] = labels
    
    return df

def handle_imbalanced_data(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print("Balanced dataset shape:", X_balanced.shape)
    return X_balanced, y_balanced

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
   
    y_pred = model.predict(X_test)
  
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
  
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_feature_importance(model, feature_names):
   
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

def main():
  
    print("Creating sample dataset...")
    data = create_sample_data(n_samples=10000)
    print("Dataset shape:", data.shape)
    print("\nClass distribution:")
    print(data['Class'].value_counts(normalize=True))
   
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
  
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
  
    print("\nBalancing the dataset using SMOTE...")
    X_train_balanced, y_train_balanced = handle_imbalanced_data(
        X_train_scaled, y_train
    )
  
    print("\nTraining the Random Forest model...")
    model = train_model(X_train_balanced, y_train_balanced)
   
    print("\nEvaluating the model...")
    evaluate_model(model, X_test_scaled, y_test)
   
    plot_feature_importance(model, X.columns)

if __name__ == "__main__":
    main()