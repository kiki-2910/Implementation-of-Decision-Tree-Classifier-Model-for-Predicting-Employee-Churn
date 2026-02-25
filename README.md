# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Required Libraries 
2. Load and Prepare the Dataset
3. Train the Decision Tree Model
4. Evaluate the Model

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S.Keerthana
RegisterNumber:  25004216
*/
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- STEP 1: Load your local dataset ---
file_path =("C:/Users/acer/Downloads/Employee.csv")
df = pd.read_csv(file_path)

# Display first few rows to verify column names
print("Dataset Columns:", df.columns.tolist())

# --- STEP 2: Preprocessing ---
# Note: Ensure the column names ('salary', 'left') match your CSV exactly.
# If your CSV uses different names, update them below.
if 'salary' in df.columns:
    salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
    df['salary'] = df['salary'].map(salary_mapping)

# Handle other categorical text data if necessary (e.g., Department)
df = pd.get_dummies(df, drop_first=True)

# Define Features (X) and Target (y)
# Replace 'left' with the actual name of your target column if different
target_col = 'left' 
X = df.drop(target_col, axis=1)
y = df[target_col]

# --- STEP 3: Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 4: Train Model ---
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)

# --- STEP 5: Evaluation & Visualization ---
y_pred = model.predict(X_test)

print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 1. Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# 2. Plot Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(model, 
          feature_names=X.columns, 
          class_names=['Stayed', 'Left'], 
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Employee Churn Decision Tree Logic")
plt.show()


```

## Output:
<img width="1240" height="52" alt="Screenshot 2026-02-25 113839" src="https://github.com/user-attachments/assets/027a623e-1a42-4735-91bb-9c9427aedf4c" />
<img width="718" height="68" alt="Screenshot 2026-02-25 113922" src="https://github.com/user-attachments/assets/b7c013bc-1659-4f07-8bf1-131023c03f00" />
<img width="695" height="276" alt="Screenshot 2026-02-25 113943" src="https://github.com/user-attachments/assets/abe28b2f-d64e-498e-9b74-9282d6299428" />
<img width="768" height="507" alt="Screenshot 2026-02-25 114013" src="https://github.com/user-attachments/assets/02e26808-3609-4fe8-84f6-81cdd23ab1db" />
<img width="1227" height="642" alt="Screenshot 2026-02-25 114040" src="https://github.com/user-attachments/assets/1814a1d0-3067-4766-a104-04bdf800ad3a" />









## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
