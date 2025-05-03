import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

#Task1
gray_img = cv2.imread('GrayScale.PNG', 0)

def show_image(title, img, wait_time=3000):
    cv2.imshow(title, img)
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()

def resize_image(img, scale_x, scale_y, interpolation):
    return cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=interpolation)

def blur_image(img, method, kernel_size=(5, 5)):
    if method == 'box':
        return cv2.blur(img, kernel_size)
    elif method == 'gaussian':
        return cv2.GaussianBlur(img, kernel_size, sigmaX=0)
    elif method == 'adaptive':
        return cv2.medianBlur(img, ksize=kernel_size[0]) 
    
resized_linear = resize_image(gray_img, 0.5, 0.5, cv2.INTER_LINEAR)
resized_nearest = resize_image(gray_img, 0.5, 0.5, cv2.INTER_NEAREST)
resized_cubic = resize_image(gray_img, 0.5, 0.5, cv2.INTER_CUBIC)  

show_image("Resized Linear", resized_linear)
show_image("Resized Nearest Neighbors", resized_nearest)
show_image("Resized Polynomial", resized_cubic)

box_blur = blur_image(gray_img, 'box', (5, 5))
gaussian_blur = blur_image(gray_img, 'gaussian', (5, 5))
adaptive_blur = blur_image(gray_img, 'adaptive', (5, 5))  

show_image("Box Blurring", box_blur)
show_image("Gaussian Blurring", gaussian_blur)
show_image("Adaptive Blurring", adaptive_blur)

#Task2
(X, y), (X_test, y_test) = mnist.load_data()
X = X.reshape(X.shape[0], -1)  # Flatten images
X_test = X_test.reshape(X_test.shape[0], -1)

X = X / 255.0  
X_test = X_test / 255.0  

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "SVM": SVC(probability=True, kernel='linear', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1) if y_prob is not None else (None, None, None)
    roc_auc = auc(fpr, tpr) if y_prob is not None else None
    
    results[model_name] = {
        "CrossVal Accuracy": np.mean(cv_scores),
        "Test Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion Matrix": conf_matrix,
        "ROC-AUC": roc_auc
    }
    
for model, metrics in results.items():
    print(f"\n{model} Results:")
    for metric, value in metrics.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value:.4f}")
    
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, (model_name, metrics) in enumerate(results.items()):
    sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f"Confusion Matrix - {model_name}")
    axes[idx].set_xlabel("Predicted Label")
    axes[idx].set_ylabel("True Label")
plt.tight_layout()
plt.show()
