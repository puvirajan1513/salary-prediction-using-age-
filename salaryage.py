import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

data={ "age": [22, 25, 28, 30, 35, 40, 45, 50, 55, 60, 22, 23, 27, 32, 38, 41, 46, 51, 56, 61],
    "education_level": [12, 14, 16, 18, 20, 12, 14, 16, 18, 20, 10, 12, 14, 16, 18, 20, 12, 14, 16, 18],
    "income_above_50k": [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]  # 1 = Income > $50K
}
df=pd.DataFrame(data)
x=df[["age","education_level"]]
y=df["income_above_50k"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=LogisticRegression()
model.fit(x_train,y_train)

pre=model.predict(x_test)

acc=accuracy_score(y_test,pre)
print(f"accuracy:{acc:.2f}")

print("classification:\n",classification_report(y_test,pre))

confm=confusion_matrix(y_test,pre)
sns.heatmap(confm,annot=True,cmap="Blues",fmt="d",xticklabels=["<=50k",">50k"],yticklabels=["<-50k",">50k"])
plt.xlabel("Pretic Lable")
plt.ylabel("True Lablel")
plt.title("confusion matrix")
plt.show()
# Predict income for a new person
new_person = np.array([[34, 16]])  # Age 34, Education Level 16
new_person_scaled = scaler.transform(new_person)

prediction = model.predict(new_person_scaled)
probability = model.predict_proba(new_person_scaled)[:, 1]  # Probability of earning >$50K

print(f"Prediction: {'>50K' if prediction[0] == 1 else '<=50K'}")
print(f"Probability of earning >$50K: {probability[0]:.2f}")


