import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

#training

df = pd.read_csv('features.csv')
le = LabelEncoder()
X = df.drop(columns=['label','Unnamed: 0'])
y = le.fit_transform(df['label'])
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=7)
model = LGBMClassifier(max_depth=15, learning_rate=0.23, path_smooth=20, max_bin=90, verbose=-1, force_col_wise=True)
model.fit(X_train,y_train)
ans = model.predict(X_test)
print(accuracy_score(y_test,ans))
joblib.dump(model, "classifier.pkl")
joblib.dump(le, 'encoder.pkl')