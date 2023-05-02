import numpy as np

from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from json import loads
from math import sqrt
from optuna import create_study
from os import environ
from pandas import DataFrame, set_option
from pyspark import SparkConf, SparkContext
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sys import argv, executable
from xgboost import XGBClassifier

set_option('display.max_columns', None)


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
        USER_FEATURE_COLS=['review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars', 'compliment_hot',
                           'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list',
                           'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny',
                           'compliment_writer', 'compliment_photos'
                           ],
        BUSINESS_FEATURE_COLS=['business_stars', 'business_review_count', 'latitude', 'longitude',
                               'business_attributes'
                               ],
        BUSINESS_BOOL_FEATURE_ATTRIBUTES=['BikeParking', 'BusinessAcceptsCreditCards', 'GoodForKids', 'HasTV',
                                          'OutdoorSeating', 'RestaurantsDelivery', 'RestaurantsGoodForGroups',
                                          'RestaurantsReservations', 'RestaurantsTakeOut'
                                          ],
        BUSINESS_CATEGORIAL_FEATURE_ATTRIBUTES=['NoiseLevel', 'RestaurantsAttire',
                                                'RestaurantsPriceRange2'
                                                ]
    )


def get_business_feature_col_extractors():
    def _attribute_extractor(business_obj):
        if business_obj['attributes'] is None:
            return tuple([None] * (
                    len(PARAMS_NS.BUSINESS_BOOL_FEATURE_ATTRIBUTES) +
                    len(PARAMS_NS.BUSINESS_CATEGORIAL_FEATURE_ATTRIBUTES)
            ))
        return tuple(map(
            lambda attr: bool(business_obj['attributes'][attr]) if attr in business_obj['attributes'] else None,
            PARAMS_NS.BUSINESS_BOOL_FEATURE_ATTRIBUTES
        )) + (
            business_obj['attributes'].get(PARAMS_NS.BUSINESS_CATEGORIAL_FEATURE_ATTRIBUTES[0], None),
            business_obj['attributes'].get(PARAMS_NS.BUSINESS_CATEGORIAL_FEATURE_ATTRIBUTES[1], None),
            business_obj['attributes'].get(PARAMS_NS.BUSINESS_CATEGORIAL_FEATURE_ATTRIBUTES[2], None),
        )

    return {
        'business_stars': lambda business_obj: (int(business_obj['stars']), ),
        'business_review_count': lambda business_obj: (int(business_obj['review_count']), ),
        'business_attributes': _attribute_extractor,
        'latitude': lambda business_obj: (business_obj.get('latitude', None), ),
        'longitude': lambda business_obj: (business_obj.get('longitude', None), ),
    }


def parse_train_set():
    filename = '{}/{}'.format(PARAMS_NS.IN_DIR, PARAMS_NS.TRAIN_FILE)
    with open(filename, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(filename) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[0], record[1], float(record[2])))


def parse_test_set():
    with open(PARAMS_NS.TEST_FILE, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(PARAMS_NS.TEST_FILE) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[0], record[1]))


def parse_val_set():
    with open(PARAMS_NS.TEST_FILE, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(PARAMS_NS.TEST_FILE) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: ((record[0], record[1]), float(record[2])))


def parse_user_set():
    filename = '{}/{}'.format(PARAMS_NS.IN_DIR, PARAMS_NS.USER_FILE)

    return sc.textFile(filename) \
        .map(lambda json_string: loads(json_string)) \
        .map(lambda user_obj: (user_obj[PARAMS_NS.RECORD_COLS[0]],
                               tuple(map(
                                   lambda col_name: user_obj[col_name],
                                   PARAMS_NS.USER_FEATURE_COLS
                               ))
                               )
             )


def parse_business_set():
    def _business_data_parser(business_obj):
        result = tuple()
        for col_name in PARAMS_NS.BUSINESS_FEATURE_COLS[: -1]:
            result += BUSINESS_FEATURE_EXTRACTORS[col_name](business_obj)
        return result

    filename = '{}/{}'.format(PARAMS_NS.IN_DIR, PARAMS_NS.BUSINESS_FILE)
    return sc.textFile(filename) \
        .map(lambda json_string: loads(json_string)) \
        .map(lambda business_obj: (business_obj[PARAMS_NS.RECORD_COLS[1]], _business_data_parser(business_obj)))


def write_results_to_file(data):
    file_header = '{}\n'.format(', '.join(PARAMS_NS.RECORD_COLS))
    with open(PARAMS_NS.OUT_FILE, 'w') as fh:
        fh.write(file_header)
        for record in data:
            fh.write('{}\n'.format(','.join(map(lambda item: str(item), list(record)))))


def fill_features(record, user_data, business_data):
    user_features = user_data.get(record[0], tuple([0] * len(PARAMS_NS.USER_FEATURE_COLS)))
    business_features = business_data.get(
        record[1],
        tuple([0] * (
                    len(PARAMS_NS.BUSINESS_FEATURE_COLS[: -1]) +
                    len(PARAMS_NS.BUSINESS_BOOL_FEATURE_ATTRIBUTES) +
                    len(PARAMS_NS.BUSINESS_CATEGORIAL_FEATURE_ATTRIBUTES)
        )))
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
    print(classification_report(list(validations.values()), list(predictions.values())))
    print(confusion_matrix(list(validations.values()), list(predictions.values())))
    print('RMSE: {}'.format(rmse))
    return f1_score(list(validations.values()), list(predictions.values()), average='weighted')


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
            'max_depth': trial.suggest_int('max_depth', 10, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 100),
            # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            # 'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            # 'subsample': trial.suggest_float('subsample', 0.01, 1.0, log=True),
            # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0, log=True),
            # 'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            # 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            # 'eval_metric': 'rmse',
            'tree_method': 'gpu_hist',
            'booster': 'gbtree',
            'verbosity': 0
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
    train_df = DataFrame(
        parse_train_set()
        .map(lambda record: fill_features(record, user_data, business_data))
        .collect(),
        columns=
        PARAMS_NS.RECORD_COLS +
        PARAMS_NS.USER_FEATURE_COLS +
        PARAMS_NS.BUSINESS_FEATURE_COLS[: -1] +
        PARAMS_NS.BUSINESS_BOOL_FEATURE_ATTRIBUTES +
        PARAMS_NS.BUSINESS_CATEGORIAL_FEATURE_ATTRIBUTES
    )
    train_df = train_df.fillna(value=np.nan)
    for col_name in PARAMS_NS.BUSINESS_BOOL_FEATURE_ATTRIBUTES:
        train_df[col_name] = LabelEncoder().fit_transform(train_df[col_name])
    for col_name in PARAMS_NS.BUSINESS_CATEGORIAL_FEATURE_ATTRIBUTES:
        train_df[col_name] = LabelEncoder().fit_transform(train_df[col_name])

    # create test dataset
    test_set = parse_test_set() \
        .map(lambda record: fill_features(record, user_data, business_data)) \
        .collect()
    test_df = DataFrame(
        test_set,
        columns=
        PARAMS_NS.RECORD_COLS[: -1] +
        PARAMS_NS.USER_FEATURE_COLS +
        PARAMS_NS.BUSINESS_FEATURE_COLS[: -1] +
        PARAMS_NS.BUSINESS_BOOL_FEATURE_ATTRIBUTES +
        PARAMS_NS.BUSINESS_CATEGORIAL_FEATURE_ATTRIBUTES
    )
    test_df = test_df.fillna(value=np.nan)
    for col_name in PARAMS_NS.BUSINESS_BOOL_FEATURE_ATTRIBUTES:
        test_df[col_name] = LabelEncoder().fit_transform(test_df[col_name])
    for col_name in PARAMS_NS.BUSINESS_CATEGORIAL_FEATURE_ATTRIBUTES:
        test_df[col_name] = LabelEncoder().fit_transform(test_df[col_name])

    # format data and write to file
    # write_results_to_file(map(
    #     lambda indexed_pair: (indexed_pair[1][0], indexed_pair[1][1], predictions[indexed_pair[0]]),
    #     enumerate(test_set)
    # ))

    x_train, y_train = train_df.drop(PARAMS_NS.RECORD_COLS, axis=1).values, \
        train_df[PARAMS_NS.RECORD_COLS[-1:]].values.ravel()
    x_test = test_df.drop(PARAMS_NS.RECORD_COLS[: -1], axis=1).values
    validations = parse_val_set().collectAsMap()

    study = create_study(direction='maximize')
    study.optimize(_objective, n_trials=1)
    log_results(study)


if __name__ == '__main__':
    # set executables
    environ['PYSPARK_PYTHON'] = executable
    environ['PYSPARK_DRIVER_PYTHON'] = executable

    # initialize program parameters
    PARAMS_NS = parse_args()
    BUSINESS_FEATURE_EXTRACTORS = get_business_feature_col_extractors()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(PARAMS_NS.APP_NAME).setMaster("local[*]"))
    sc.setLogLevel('ERROR')

    # run prediction
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
