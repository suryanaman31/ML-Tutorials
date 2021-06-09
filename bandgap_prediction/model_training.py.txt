import scipy
import matplotlib
import numpy
import sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import pickle

train_path = "C:\\Users\\91775\\OneDrive\\Desktop\\ML_and_DL\\bandgap_prediction\\train_data.csv"
test_path = "C:\\Users\\91775\\OneDrive\\Desktop\\ML_and_DL\\bandgap_prediction\\test_data.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

p=[]
for i in range(6,82):
    p.append(i)
X = train_data.iloc[:, p]  
y = train_data.iloc[:, 1]
X_infer = test_data.iloc[:, p]
y_infer = test_data.iloc[:, 1]

model = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                    max_samples=None, min_impurity_decrease=0.0,
                    min_impurity_split=None, min_samples_leaf=1,
                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                    n_estimators=100, n_jobs=-1, oob_score=False,
                    random_state=123, verbose=0, warm_start=False)

model.fit(X, y)
   
#pickle_out = open("model.pkl","wb")
#pickle.dump(model, pickle_out)
#pickle_out.close()
