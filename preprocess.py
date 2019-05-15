import pandas as pd
import numpy as np
from mdlp.discretization import MDLP


TITLE_LIST = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr', 'Ms',
              'Mlle', 'Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer']

CABIN_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'U']


def substring_exist(string, substrings):
    for substring in substrings:
        if str.find(string, substring) != -1:
            return substring
    print(string)
    return np.nan


def replace_titles(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        # return 'Mr'
        return 0
    elif title in ['Countess', 'Mme']:
        # return 'Mrs'
        return 0.75
    elif title in ['Mlle', 'Ms']:
        # return 'Miss'
        return 1
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            # return 'Mr'
            return 0
        else:
            # return 'Mrs'
            return 0.75
    else:
        # return title
        return 0.5


def replace_sex(x):
    sex = x['Sex']
    if sex == "male":
        return 0
    else:
        return 1


def replace_embark(x):
    embarked = x['Embarked']
    if embarked == "S":
        return 0
    elif embarked == "Q":
        return 1
    else:
        return -1


def replace_deck(x):
    deck = x['Deck']
    if deck == "U":
        return 0
    elif deck == "G":
        return 0.15
    elif deck == "F":
        return 0.3
    elif deck == "E":
        return 0.45
    elif deck == "D":
        return 0.6
    elif deck == "C":
        return 0.75
    elif deck == "B":
        return 0.9
    elif deck == "A":
        return 1.1
    elif deck == "T":
        return 1.2
    else:
        return -1


df = pd.read_csv("data/train.csv")
df = df.drop(['PassengerId', 'Ticket'], axis=1)

age_mean = df['Age'].mean()
embark_mode = (df["Embarked"].mode().values[0])


df['Age'] = df['Age'].fillna(age_mean)
df['Embarked'] = df['Embarked'].fillna(embark_mode)
df['Cabin'] = df['Cabin'].fillna("U")

df['Title'] = df['Name'].map(
    lambda x: substring_exist(x, TITLE_LIST))


df['Title'] = df.apply(replace_titles, axis=1)
df['Embarked'] = df.apply(replace_embark, axis=1)
df['Deck'] = df['Cabin'].map(lambda x: substring_exist(x, CABIN_LIST))
df['Deck'] = df.apply(replace_deck, axis=1)

df['Family_Size'] = df['SibSp']+df['Parch']

df['Fare_Per_Person'] = df['Fare']/(df['Family_Size']+1)

transformer = MDLP()
X_age = df["Age"].to_numpy().reshape((df["Age"].shape[0], 1))
y_age = df["Survived"].to_numpy().reshape((df["Survived"].shape[0], 1))

disc = transformer.fit_transform(X_age, y_age)
df["Age_disc"] = disc

df['Sex'] = df.apply(replace_sex, axis=1)

df["Pclass"] = df["Pclass"].map(lambda x: 1/x)

df.to_csv("./data/train_neat.csv")

print(df.columns)
