# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import numpy as np

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data             # Feature matrix
y = iris.target           # Labels (0: setosa, 1: versicolor, 2: virginica)
target_names = iris.target_names

# 2. Check for missing values (dataset is known to be clean, but good to verify)
if np.isnan(X).any():
    
    print("\n😥Missing values detected!")
else:
    print("\n😋😎 No missing values in the dataset.")

# 3. Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(
    criterion='gini',      # Can also try 'entropy'
    max_depth=3,           # Optional: limits depth to reduce overfitting
    random_state=42
)
clf.fit(X_train, y_train)

# 5. Predict on the test set
y_pred = clf.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)

print("")
print("\nEvaluation Results [All eyes on me kam 2Pac😊]: ")
print("________________________________________________________")
print(f"🎯 Accuracy: {accuracy:.2f}")

# Optional: print full classification report
print("\n📜 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\n Built By Charles for AI for SE Week 3 Practical😁")
