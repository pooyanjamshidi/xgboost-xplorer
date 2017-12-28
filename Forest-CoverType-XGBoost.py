# import classes and functions
import numpy
from pandas import read_csv
from xgboost import XGBClassifier  # xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import time

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataframe = read_csv("data/covtype.data", header=None)
dataset = dataframe.values

# reshuffle dataset
dataset = numpy.random.permutation(dataset)

# use reduced dataset
dataset = dataset[0:50000,:]

# split into input (X) and output (Y) variables
X = dataset[:,0:54].astype(float)
Y = dataset[:,54]

# encode class values as integers
encoder = LabelEncoder()
encoder = encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# XGBOOST
# grid search
model = XGBClassifier(nthread=3, n_estimators=50, max_depth=3, learning_rate=0.01)

start = time.time()
results = model.fit(X, encoded_Y, eval_set=(X, encoded_Y))
end = time.time()
inference_time = end - start

print(inference_time, results.evals_result_)
