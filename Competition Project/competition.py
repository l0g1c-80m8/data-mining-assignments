from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from json import loads
from math import sqrt
from optuna import create_study
from os import environ
from pandas import DataFrame
from pyspark import SparkConf, SparkContext
from sys import argv, executable
from xgboost import XGBRegressor


def parse_args():
    if len(argv) < 3:
        # expected arguments: script path, dataset path, output file path
        print('ERR: Expected two arguments: (input file path, output file path).')
        exit(1)

    # read program arguments
    return Namespace(
        APP_NAME='competition_project',
        IN_DIR=argv[1],
        TEST_FILE=argv[2],
        OUT_FILE=argv[3],
        TRAIN_FILE='yelp_train.csv',
        USER_FILE='user.json',
        BUSINESS_FILE='business.json',
        RECORD_COLS=['user_id', 'business_id', 'rating'],
        USER_FEATURE_COLS=['review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars'],
        BUSINESS_FEATURE_COLS=['business_stars', 'business_review_count'],
    )


def parse_train_set():
    filename = '{}/{}'.format(PARAMS.IN_DIR, PARAMS.TRAIN_FILE)
    with open(filename, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(filename) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[0], record[1], float(record[2])))


def parse_test_set():
    with open(PARAMS.TEST_FILE, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(PARAMS.TEST_FILE) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[0], record[1]))


def parse_val_set():
    with open(PARAMS.TEST_FILE, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(PARAMS.TEST_FILE) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: ((record[0], record[1]), float(record[2])))


def parse_user_set():
    filename = '{}/{}'.format(PARAMS.IN_DIR, PARAMS.USER_FILE)

    return sc.textFile(filename) \
        .map(lambda json_string: loads(json_string)) \
        .map(lambda user_obj: (user_obj[PARAMS.RECORD_COLS[0]],
                               tuple(map(
                                   lambda col_name: user_obj[col_name],
                                   PARAMS.USER_FEATURE_COLS
                               ))
                               )
             )


def parse_business_set():
    filename = '{}/{}'.format(PARAMS.IN_DIR, PARAMS.BUSINESS_FILE)

    return sc.textFile(filename) \
        .map(lambda json_string: loads(json_string)) \
        .map(lambda business_obj: (business_obj[PARAMS.RECORD_COLS[1]],
                                   tuple(map(
                                       lambda col_name: business_obj[col_name.replace('business_', '')],
                                       PARAMS.BUSINESS_FEATURE_COLS
                                   ))
                                   )
             )


def write_results_to_file(data):
    file_header = '{}\n'.format(', '.join(PARAMS.RECORD_COLS))
    with open(PARAMS.OUT_FILE, 'w') as fh:
        fh.write(file_header)
        for record in data:
            fh.write('{}\n'.format(','.join(map(lambda item: str(item), list(record)))))


def fill_features(record, user_data, business_data):
    user_features = user_data.get(record[0], tuple([0] * len(PARAMS.USER_FEATURE_COLS)))
    business_features = business_data.get(record[1], tuple([0] * len(PARAMS.BUSINESS_FEATURE_COLS)))
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

    for key, value in trial.PARAMS.items():
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
            'use_label_encoder': False,
            'tree_method': 'gpu_hist',
            'booster': 'gbtree',
            'verbosity': 0
        }

        # Fit the model
        optuna_model = XGBRegressor(**hyper_params)
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
    train_df = DataFrame(
        parse_train_set()
        .map(lambda record: fill_features(record, user_data, business_data))
        .collect(),
        columns=PARAMS.RECORD_COLS + PARAMS.USER_FEATURE_COLS + PARAMS.BUSINESS_FEATURE_COLS
    )

    # create test dataset
    test_set = parse_test_set() \
        .map(lambda record: fill_features(record, user_data, business_data)) \
        .collect()
    test_df = DataFrame(test_set,
                           columns=PARAMS.RECORD_COLS[: -1] + PARAMS.USER_FEATURE_COLS + PARAMS[
                               'business_feature_cols']
                           )

    # format data and write to file
    # write_results_to_file(map(
    #     lambda indexed_pair: (indexed_pair[1][0], indexed_pair[1][1], predictions[indexed_pair[0]]),
    #     enumerate(test_set)
    # ))

    x_train, y_train = train_df.drop(PARAMS.RECORD_COLS, axis=1).values, \
        train_df[PARAMS.RECORD_COLS[-1:]].values
    x_test = test_df.drop(PARAMS.RECORD_COLS[: -1], axis=1).values
    validations = parse_val_set().collectAsMap()

    study = create_study(direction='minimize')
    study.optimize(_objective, n_trials=1)

    log_results(study)


if __name__ == '__main__':
    # set executables
    environ['PYSPARK_PYTHON'] = executable
    environ['PYSPARK_DRIVER_PYTHON'] = executable

    # initialize program parameters
    PARAMS = parse_args()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(PARAMS.APP_NAME).setMaster("local[*]"))
    sc.setLogLevel('ERROR')

    # run prediction
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
