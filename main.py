import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
dataset = pd.read_csv(SCRIPT_PATH + "/train.csv")

#features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
training_data = dataset[features]
# print("===== training_data =====")
# print(training_data)
training_label = dataset['Survived']
# print("===== training_label =====")
# print(training_label)

# training_data = training_data.fillna(0)

training_data['Sex'] = LabelEncoder().fit_transform(training_data['Sex'].fillna('0'))
training_data['Cabin'] = LabelEncoder().fit_transform(training_data['Cabin'].fillna('0'))
training_data['Embarked'] = LabelEncoder().fit_transform(training_data['Embarked'].fillna('0'))

training_data['Age'] = training_data['Age'].fillna('0')
# print("===== training_data =====")
# print(training_data)

model = RandomForestClassifier()
model.fit(training_data, training_label)

y_pos = np.arange(len(features))
plt.barh(y_pos, model.feature_importances_, align='center', alpha=0.4)
plt.yticks(y_pos, features)
plt.xlabel('features')
plt.title('feature_importances')
plt.show()