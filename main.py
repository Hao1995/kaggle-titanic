import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
dataset = pd.read_csv(SCRIPT_PATH + "/train.csv")

# ===== Data analysis
#features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
training_data = dataset[features]
# print("===== training_data =====")
# print(training_data)
training_label = dataset['Survived']

# training_data = training_data.fillna(0)
training_data['Sex'] = LabelEncoder().fit_transform(training_data['Sex'].fillna('0'))
training_data['Cabin'] = LabelEncoder().fit_transform(training_data['Cabin'].fillna('0'))
training_data['Embarked'] = LabelEncoder().fit_transform(training_data['Embarked'].fillna('0'))
training_data['Age'] = training_data['Age'].fillna('0')

model = RandomForestClassifier()
model.fit(training_data, training_label)

y_pos = np.arange(len(features))
plt.barh(y_pos, model.feature_importances_, align='center', alpha=0.4)
plt.yticks(y_pos, features)
plt.xlabel('features')
plt.title('feature_importances')
plt.show()


# === Prepare to train data
features = ['Sex', 'Age', 'Fare']
training_data = dataset[features]

training_data['Sex'] = LabelEncoder().fit_transform(training_data['Sex'].fillna('0'))
training_data['Age'] = training_data['Age'].fillna(0)
model = RandomForestClassifier()
model.fit(training_data, training_label)

test_data = pd.read_csv(SCRIPT_PATH + "/test.csv")
prediction_data = test_data[features]

prediction_data['Sex'] = LabelEncoder().fit_transform(prediction_data['Sex'].fillna('0'))
prediction_data['Fare'] = prediction_data['Sex'].fillna('0')
prediction_data['Age'] = prediction_data['Age'].fillna(0)

result_lables = model.predict(prediction_data)
results = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : result_lables
})

results.to_csv(SCRIPT_PATH + "/submission.csv", index=False)
from sklearn.model_selection import cross_val_score
cv_scores = np.mean(cross_val_score(model, training_data, training_label, scoring='roc_auc', cv=5))
print(cv_scores)