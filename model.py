import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv('depression.csv')

# Split the data into features (X) and labels (y)
X = data.drop('clean_text', axis=1)
y = data['is_depression']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the SVM model
svm = SVC(kernel='linear', C=1)

# Train the SVM model
svm.fit(X_train, y_train)

# Evaluate the SVM model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion matrix:', confusion_mat)
