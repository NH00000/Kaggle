import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree

titanic = pd.read_csv('data/train.csv')
X = titanic[['pclass','age','sex']]
y = titanic['survived']