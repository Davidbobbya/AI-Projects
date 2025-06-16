import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

df = pd.read_csv("diabetes.csv")

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())

x = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=34)

model = KNeighborsClassifier(n_neighbors=6)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
print("Accuracy Score: ",accuracy_score(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))
print("Classification report",classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(data=cm, annot=True, fmt='d', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()