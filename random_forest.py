import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("./data/train_neat.csv")


X = data[["Pclass", "Sex", "Age", "SibSp",
          "Parch", "Fare", "Embarked", "Title",
          "Deck", "Family_Size", "Fare_Per_Person", "Age_disc"]]
y = data["Survived"]


rf = RandomForestClassifier(n_jobs=-1)
rf_model = rf.fit(X, y)

features = sorted(zip(rf_model.feature_importances_,
                      X.columns), reverse=True)
print(features)
