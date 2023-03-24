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


def main():
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

    # define the regressor model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, booster='gbtree', verbosity=0)
    # train the model
    model.fit(train_df.drop(params['record_cols'], axis=1).values, train_df[params['record_cols'][-1:]].values)
    # generate predictions
    predictions = model.predict(test_df.drop(params['record_cols'][: -1], axis=1).values)

    # format data and write to file
    write_results_to_file(map(
        lambda indexed_pair: (indexed_pair[1][0], indexed_pair[1][1], predictions[indexed_pair[0]]),
        enumerate(test_set)
    ))


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
