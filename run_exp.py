# import classes and functions
import numpy as np
import itertools
import csv
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import os


options = {
'min_child_weight': {1, 10},
'nthread': {1, 3},
'n_estimators': {100, 200},
'max_depth': {1, 5},
'learning_rate': {0.0001, 0.1},
'max_delta_step': {0, 10},
'subsample': {0.5, 1},
'colsample_bytree': {0.5, 1},
'lambda': {0, 1},
'alpha': {0, 1},
'scale_pos_weight': {0.1, 1}
}

options_idx = {
'min_child_weight': 0,
'nthread': 1,
'n_estimators': 2,
'max_depth': 3,
'learning_rate': 4,
'max_delta_step': 5,
'subsample': 6,
'colsample_bytree': 7,
'lambda': 8,
'alpha': 9,
'scale_pos_weight': 10
}


# confg stuff, later to be moved to config.py
test_size = 0.33
data_perc = 0.1
dataset_name = "covtype.data"
exp_path = "experiments"
data_path = "data"
num_params = len(options)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

dataframe = read_csv(os.path.join(data_path, dataset_name), header=0)
dataset = dataframe.values

# reshuffle dataset
dataset = np.random.permutation(dataset)

# use reduced dataset
data_size = dataset.shape[0]
num_attr = dataset.shape[1]
dataset = dataset[0:int(data_size*data_perc), :]

# split into input (X) and output (Y) variables
X = dataset[:, 0: num_attr - 1].astype(float)
Y = dataset[:, num_attr - 1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


confs = itertools.product(range(2), repeat=num_params)
configs = np.zeros(shape=(2**num_params, num_params))
i = 0
for c in confs:
    configs[i, :] = np.array(c)
    i += 1


for i in range(len(configs)):

    model = XGBClassifier(
        min_child_weight=list(options['min_child_weight'])[int(configs[i, options_idx['min_child_weight']])],
        nthread=list(options['nthread'])[int(configs[i, options_idx['nthread']])],
        n_estimators=list(options['n_estimators'])[int(configs[i, options_idx['n_estimators']])],
        max_depth=list(options['max_depth'])[int(configs[i, options_idx['max_depth']])],
        learning_rate=list(options['learning_rate'])[int(configs[i, options_idx['learning_rate']])],
        max_delta_step=list(options['max_delta_step'])[int(configs[i, options_idx['max_delta_step']])],
        subsample=list(options['subsample'])[int(configs[i, options_idx['subsample']])],
        colsample_bytree=list(options['colsample_bytree'])[int(configs[i, options_idx['colsample_bytree']])],
        reg_alpha=list(options['alpha'])[int(configs[i, options_idx['alpha']])],
        reg_lambda=list(options['lambda'])[int(configs[i, options_idx['lambda']])],
        scale_pos_weight=list(options['scale_pos_weight'])[int(configs[i, options_idx['scale_pos_weight']])]
    )

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_time = end - start

    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    test_time = end-start

    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)

    measurement = {'config_id': i, 'train_time': train_time, 'test_time': test_time, 'accuracy': accuracy * 100}

    csv_filename = 'exp_' + dataset_name + '.csv'
    myField = ["config_id", "train_time", "test_time", "accuracy"]

    with open(os.path.join("experiments", csv_filename), 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=myField)
        writer.writerow(measurement)
