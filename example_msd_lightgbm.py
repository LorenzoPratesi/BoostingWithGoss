import numpy as np
import time
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pandas as pd


def get_fold(data):
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    n_training = 463715
    X_train, X_test, y_train, y_test = X[:n_training], X[n_training:], y[:n_training], y[n_training:]
    return X_train, X_test, y_train, y_test



params = {'min_split_gain': 0,
          'min_data_in_leaf': 1,
          'max_depth': -1,
          'max_bin': 64,
          'learning_rate': 0.1,
          'n_estimators': 200,
          'verbose': 2,
          'feature_fraction': 1,
          'bagging_fraction': 1,
          'seed': 1}

# %% LightGBM specific
method = 'lightgbm'
params['lambda'] = 1
params['device'] = 'cpu'
params['objective'] = 'regression'
params['metric'] = 'rmse'
params['num_leaves'] = 8
params['bagging_freq'] = 1
params['min_data_in_bin'] = 1
# %% Loop
datasets = ['msd']
base_estimators = 1000
df = pd.DataFrame(
    columns=['method', 'dataset', 'fold', 'device', 'validation_estimators', 'test_estimators', 'rmse_test',
             'crps_test', 'validation_time'])

for i, dataset in enumerate(datasets):
    params['bagging_fraction'] = 0.1
    n_folds = 1

    data = pd.read_feather('../datasets/msd.feather')
    for fold in range(n_folds):
        print(f'{dataset}: fold {fold + 1}/{n_folds}')

        # Get data
        X_train, X_test, y_train, y_test = get_fold(data)
        X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=fold)

        # Build torchdata datasets
        train_data = lgb.Dataset(X_train, y_train)
        train_val_data = lgb.Dataset(X_train_val, y_train_val)
        valid_data = lgb.Dataset(X_val, y_val)
        test_data = lgb.Dataset(X_test, y_test)
        params['n_estimators'] = base_estimators

        # Train to retrieve best iteration
        print('Validating...')
        start = time.perf_counter()
        model = lgb.train(params, train_val_data, valid_sets=valid_data, early_stopping_rounds=2000)
        end = time.perf_counter()
        validation_time = end - start
        print(f'Fold time: {validation_time:.2f}s')

        # Set iterations to best iteration
        params['n_estimators'] = model.best_iteration + 1

        # Retrain on full set
        print('Training...')
        model = lgb.train(params, train_data)

        # % Predictions
        print('Prediction...')
        yhat_point = model.predict(X_test)

        # Scoring
        rmse = np.sqrt(mean_squared_error(y_test, yhat_point))
        print("RMSE: ", rmse)
        print("R2 coefficient:", r2_score(y_test, yhat_point))
        print("Mean Absolute Error:", mean_absolute_error(y_test, yhat_point))
        crps = 0

        # Save data
        df = df.append({'method': method, 'dataset': dataset, 'fold': fold, 'device': params['device'],
                        'validation_estimators': base_estimators, 'test_estimators': params['n_estimators'],
                        'rmse_test': rmse, 'crps_test': crps, 'validation_time': validation_time}, ignore_index=True)
# %% Save
filename = f"{method}_{params['device']}.csv"
df.to_csv(f'{filename}')
