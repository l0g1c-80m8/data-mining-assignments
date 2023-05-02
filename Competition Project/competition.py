import json
import pandas as pd
import os
import optuna
import sys

from collections import defaultdict
from datetime import datetime
from math import sqrt
from pyspark import SparkConf, SparkContext
from xgboost import XGBClassifier


def parse_args():
    if len(sys.argv) < 3:
        # expected arguments: script path, dataset path, output file path
        print('ERR: Expected two arguments: (input file path, output file path).')
        exit(1)

    # read program arguments
    run_time_params = dict()
    run_time_params['app_name'] = 'hw3-task2_2'
    run_time_params['in_dir'] = sys.argv[1]
    run_time_params['test_file'] = sys.argv[2]
    run_time_params['out_file'] = sys.argv[3]
    run_time_params['train'] = 'yelp_train.csv'
    run_time_params['user'] = 'user.json'
    run_time_params['business'] = 'business.json'
    run_time_params['record_cols'] = ['user_id', 'business_id', 'rating']
    run_time_params['user_feature_cols'] = ['review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars']
    run_time_params['business_feature_cols'] = ['business_stars', 'business_review_count']
    return run_time_params


def parse_train_set():
    filename = '{}/{}'.format(params['in_dir'], params['train'])
    with open(filename, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(filename) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[0], record[1], float(record[2])))


def parse_test_set():
    with open(params['test_file'], 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(params['test_file']) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[0], record[1]))


def parse_val_set():
    with open(params['test_file'], 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(params['test_file']) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: ((record[0], record[1]), float(record[2])))


def parse_user_set():
    filename = '{}/{}'.format(params['in_dir'], params['user'])

    return sc.textFile(filename) \
        .map(lambda json_string: json.loads(json_string)) \
        .map(lambda user_obj: (user_obj[params['record_cols'][0]],
                               tuple(map(
                                   lambda col_name: user_obj[col_name],
                                   params['user_feature_cols']
                               ))
                               )
             )


def parse_business_set():
    filename = '{}/{}'.format(params['in_dir'], params['business'])

    return sc.textFile(filename) \
        .map(lambda json_string: json.loads(json_string)) \
        .map(lambda business_obj: (business_obj[params['record_cols'][1]],
                                   tuple(map(
                                       lambda col_name: business_obj[col_name.replace('business_', '')],
                                       params['business_feature_cols']
                                   ))
                                   )
             )


def write_results_to_file(data):
    file_header = '{}\n'.format(', '.join(params['record_cols']))
    with open(params['out_file'], 'w') as fh:
        fh.write(file_header)
        for record in data:
            fh.write('{}\n'.format(','.join(map(lambda item: str(item), list(record)))))


def fill_features(record, user_data, business_data):
    user_features = user_data.get(record[0], tuple([0] * len(params['user_feature_cols'])))
    business_features = business_data.get(record[1], tuple([0] * len(params['business_feature_cols'])))
    return record + user_features + business_features


def evaluate(validations, predictions):
    rmse = 0.0
    distribution = defaultdict(int)
    for key in predictions.keys():
        diff = abs(predictions[key] - validations[key])
        if diff < 0:
            distribution['<0'] += 1
        elif diff < 1:
            distribution['<1'] += 1
        elif diff < 2:
            distribution['<2'] += 1
        elif diff < 3:
            distribution['<3'] += 1
        elif diff < 4:
            distribution['<4'] += 1
        elif diff <= 5:
            distribution['<=5'] += 1
        else:
            distribution['>5'] += 1

        rmse += pow(diff, 2)

    rmse = sqrt(rmse / sum(distribution.values()))

    # print(distribution, sum(distribution.values()))
    return rmse


def log_results(study):
    print('Number of finished trials: {}'.format(len(study.trials)))
    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))
    print('  Params: ')

    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


def main():
    def _objective(trial):
        hyper_params = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 5, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'eval_metric': 'mlogloss',
            'use_label_encoder': False
        }

        # Fit the model
        optuna_model = XGBClassifier(**hyper_params)
        optuna_model.fit(x_train, y_train)

        # Make predictions
        y_pred = optuna_model.predict(x_test)

        # Evaluate predictions
        accuracy = evaluate(validations, dict(map(
            lambda indexed_pair: ((indexed_pair[1][0], indexed_pair[1][1]), y_pred[indexed_pair[0]]),
            enumerate(test_set)
        )))
        return accuracy

    # create feature data
    user_data = parse_user_set().collectAsMap()
    business_data = parse_business_set().collectAsMap()

    # create training dataset
    train_df = pd.DataFrame(
        parse_train_set()
        .map(lambda record: fill_features(record, user_data, business_data))
        .collect(),
        columns=params['record_cols'] + params['user_feature_cols'] + params['business_feature_cols']
    )

    # create test dataset
    test_set = parse_test_set() \
        .map(lambda record: fill_features(record, user_data, business_data)) \
        .collect()
    test_df = pd.DataFrame(test_set,
                           columns=params['record_cols'][: -1] + params['user_feature_cols'] + params[
                               'business_feature_cols']
                           )

    # format data and write to file
    # write_results_to_file(map(
    #     lambda indexed_pair: (indexed_pair[1][0], indexed_pair[1][1], predictions[indexed_pair[0]]),
    #     enumerate(test_set)
    # ))

    x_train, y_train = train_df.drop(params['record_cols'], axis=1).values, \
        train_df[params['record_cols'][-1:]].values.ravel()
    x_test = test_df.drop(params['record_cols'][: -1], axis=1).values
    validations = parse_val_set().collectAsMap()

    study = optuna.create_study(direction='minimize')
    study.optimize(_objective, n_trials=10)

    log_results(study)


if __name__ == '__main__':
    # set executables
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # initialize program parameters
    params = parse_args()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(params['app_name']).setMaster("local[*]"))
    sc.setLogLevel('ERROR')

    # run prediction
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
