import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
dataset = pd.read_csv(SCRIPT_PATH + "/train.csv")

# ===== Data analysis
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
# features = ['Sex', 'Age', 'Fare']
training_data = dataset[features]

training_label = dataset['Survived']

training_data['Pclass'] = LabelEncoder().fit_transform(training_data['Pclass'].fillna('0'))
training_data['Sex'] = LabelEncoder().fit_transform(training_data['Sex'].fillna('0'))
training_data['Age'] = training_data['Age'].fillna(0)
training_data['SibSp'] = LabelEncoder().fit_transform(training_data['SibSp'].fillna('0'))
training_data['Parch'] = LabelEncoder().fit_transform(training_data['Parch'].fillna('0'))
training_data['Fare'] = training_data['Sex'].fillna(0)
training_data['Cabin'] = LabelEncoder().fit_transform(training_data['Cabin'].fillna('0'))
training_data['Embarked'] = LabelEncoder().fit_transform(training_data['Embarked'].fillna('0'))

from sklearn import tree

#訓練
clf = tree.DecisionTreeClassifier()
clf.fit(training_data, training_label)

from sklearn.model_selection import cross_val_score
cv_scores = np.mean(cross_val_score(clf, training_data, training_label, scoring='roc_auc', cv=5))
print(cv_scores)

# === Prediction_data
test_data = pd.read_csv(SCRIPT_PATH + "/test.csv")
prediction_data = test_data[features]

prediction_data['Pclass'] = LabelEncoder().fit_transform(prediction_data['Pclass'].fillna('0'))
prediction_data['Sex'] = LabelEncoder().fit_transform(prediction_data['Sex'].fillna('0'))
prediction_data['Age'] = prediction_data['Age'].fillna(0)
prediction_data['SibSp'] = LabelEncoder().fit_transform(prediction_data['SibSp'].fillna('0'))
prediction_data['Parch'] = LabelEncoder().fit_transform(prediction_data['Parch'].fillna('0'))
prediction_data['Fare'] = prediction_data['Sex'].fillna(0)
prediction_data['Cabin'] = LabelEncoder().fit_transform(prediction_data['Cabin'].fillna('0'))
prediction_data['Embarked'] = LabelEncoder().fit_transform(prediction_data['Embarked'].fillna('0'))

result_lables = clf.predict(prediction_data)

results = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : result_lables
})

results.to_csv(SCRIPT_PATH + "/submission/submission-" + os.path.splitext(os.path.basename(__file__))[0] + ".csv", index=False)