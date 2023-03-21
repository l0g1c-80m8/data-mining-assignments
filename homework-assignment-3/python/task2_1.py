"""
Homework Assignment 3
Task 2 (Part 1)

An item-based collaborative filtering mechanism to generate recommendation of businesses (items) for users.
"""

import os
import sys

from datetime import datetime
from functools import reduce
from math import sqrt
from pyspark import SparkConf, SparkContext


def parse_args():
    if len(sys.argv) < 3:
        # expected arguments: script path, dataset path, output file path
        print('ERR: Expected two arguments: (input file path, output file path).')
        exit(1)

    # read program arguments
    run_time_params = dict()
    run_time_params['app_name'] = 'hw3-task2_1'
    run_time_params['in_file'] = sys.argv[1]
    run_time_params['test_file'] = sys.argv[2]
    run_time_params['out_file'] = sys.argv[3]
    run_time_params['top_candidates'] = 15
    return run_time_params


def parse_dataset(filename):
    with open(filename, 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(filename) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[1], (record[0], record[2]))) \
        .groupByKey() \
        .map(lambda business_set: (business_set[0], dict(business_set[1])))


def write_results_to_file(recommendations):
    file_header = 'user_id, business_id, prediction\n'
    with open(params['out_file'], 'w') as fh:
        fh.write(file_header)
        for triple in recommendations:
            fh.write('{},{},{}\n'.format(triple[1], triple[0], triple[2]))


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
    # -------------------- START -------------------- #
    # dataset rdd
    dataset_rdd = parse_dataset(params['in_file'])
    dataset = dataset_rdd.collectAsMap()
    # test rdd
    test_rdd = parse_dataset(params['test_file']) \
        .flatMapValues(lambda val: val)

    write_results_to_file(test_rdd.collect())
    # -------------------- END -------------------- #
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
