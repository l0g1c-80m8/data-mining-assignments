import json
import pandas as pd
import os
import sys
import xgboost as xgb

from datetime import datetime
from pyspark import SparkConf, SparkContext


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


def parse_user_set():
    filename = '{}/{}'.format(params['in_dir'], params['user'])

    return sc.textFile(filename) \
        .map(lambda json_string: json.loads(json_string)) \
        .map(lambda user_obj: (user_obj['user_id'], (user_obj['review_count'],
                                                     user_obj['useful'], user_obj['funny'], user_obj['cool'],
                                                     user_obj['fans'],
                                                     user_obj['average_stars']
                                                     ))
             )


def parse_business_set():
    filename = '{}/{}'.format(params['in_dir'], params['business'])

    return sc.textFile(filename) \
        .map(lambda json_string: json.loads(json_string)) \
        .map(lambda business_obj: (business_obj['business_id'], (business_obj['stars'], business_obj['review_count'])))


def fill_features(record, user_data, business_data):
    user_features = user_data.get(record[0], (0, 0, 0, 0, 0, 0, 0))
    business_features = business_data.get(record[1], (0, 0))
    return record + user_features + business_features


def main():
    # create feature data
    user_data = parse_user_set().collectAsMap()
    business_data = parse_business_set().collectAsMap()

    # create training dataset
    train_df = pd.DataFrame(
        parse_train_set()
        .map(lambda record: fill_features(record, user_data, business_data))
        .collect(),
        columns=['user_id', 'business_id', 'rating', 'review_count', 'useful',
                 'funny', 'cool', 'fans', 'average_stars', 'business_stars',
                 'business_review_count'
                 ]
    )

    # create test dataset
    test_df = pd.DataFrame(
        parse_test_set()
        .map(lambda record: fill_features(record, user_data, business_data))
        .collect(),
        columns=['user_id', 'business_id', 'review_count', 'useful', 'funny', 'cool',
                 'fans', 'average_stars', 'business_stars', 'business_review_count'
                 ]
    )

    # define the regressor model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, verbosity=0)
    # train the model
    model.fit(train_df.drop(['user_id', 'business_id', 'rating'], axis=1).values, train_df[['rating']].values)
    # generate predictions
    predictions = model.predict(test_df.drop(['user_id', 'business_id'], axis=1).values)

    # format data and write to file
    results_df = test_df.copy(deep=True)
    results_df['ratings'] = predictions
    results_df[['user_id', 'business_id', 'ratings']].to_csv(params['out_file'], index=False)


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
