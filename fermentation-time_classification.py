# Import modules in standard order
import pandas as pd
import numpy as np
import joblib
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, chi2, f_classif, SelectFromModel, SelectPercentile
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

# Read data
vec = pd.read_csv('train.csv')
vec_label = np.loadtxt('train_label.csv', delimiter=',')

# Randomize dataset
num_samples = len(vec)
random_indices = np.random.permutation(num_samples)
shuffled_vec = vec.iloc[random_indices]
shuffled_vec_label = vec_label[random_indices]

# Feature selection pipeline
feature_selection_pipeline = Pipeline([
    ('vt', VarianceThreshold()),
    ('skb2', SelectPercentile(chi2, percentile=4.32)),
    ('skb1', SelectPercentile(f_classif, percentile=50)),
    ('sfm', SelectFromModel(svm.SVC(kernel='linear', C=0.15), threshold=0.0565))
])

# Apply feature selection
vec_selected = feature_selection_pipeline.fit_transform(shuffled_vec, shuffled_vec_label)

# SVM classifier
svm_classifier = Pipeline([
    ('classifier', svm.SVC(kernel='rbf', C=21.224536734693874, gamma=0.05022481203007519))
])

# Cross-validation
scores = cross_val_score(svm_classifier, vec_selected, shuffled_vec_label, cv=10, scoring='roc_auc', n_jobs=-1)
scores1 = cross_val_score(svm_classifier, vec_selected, shuffled_vec_label, cv=10, scoring='accuracy', n_jobs=-1)
# Print the mean Area Under the ROC Curve (AUC) score obtained from 10-fold cross-validation on the training dataset
print("Mean Area Under the ROC Curve (AUC) from 10-fold cross-validation on the training dataset:", scores.mean())

# Print the mean accuracy score obtained from 10-fold cross-validation on the training dataset
print("Mean Accuracy from 10-fold cross-validation on the training dataset:", scores1.mean())

# Train model and save
svm_classifier.fit(vec_selected, shuffled_vec_label)
joblib.dump(svm_classifier, 'svm_classifier.pkl')
joblib.dump(feature_selection_pipeline, 'feature_selection_pipeline.pkl')

# Load test dataset and make predictions
Pvec = np.loadtxt(open("test.csv", "rb"), delimiter=",")
selecter = joblib.load('feature_selection_pipeline.pkl')
selected_features = selecter.transform(Pvec)
loaded_model = joblib.load('svm_classifier.pkl')
predicted_labels = loaded_model.predict(selected_features)

# Calculate accuracy
Pvec_label = pd.read_csv('test_label.csv')
accuracy = accuracy_score(Pvec_label, predicted_labels)
print(f'Accuracy of the model on the test dataset: {accuracy}')