from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

(X, y), (X_test, y_test) = mnist.load_data()

X = X.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
rf_model = RandomForestClassifier(random_state=42,n_jobs=-1,n_estimators=100,max_features='sqrt',min_samples_leaf=1)

models = {'Naïve Bayes': nb_model, 'Random Forest': rf_model}

for name, model in models.items():
    print(f"\n{name} Model Results:")
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_valid)
    
    accuracy = np.mean(y_pred == y_valid)
    print(f"Accuracy: {accuracy:.4f}")

    print(f"Classification Report:\n{classification_report(y_valid, y_pred)}")

    conf_matrix = confusion_matrix(y_valid, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()