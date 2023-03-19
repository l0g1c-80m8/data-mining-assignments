"""
Homework Assignment 3
Task 2 (Part 1)

An item-based collaborative filtering mechanism to generate recommendation of businesses (items) for users.
"""

import os
import sys

from datetime import datetime
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
    return run_time_params


def parse_dataset():
    with open(params['in_file'], 'r') as fh:
        header = fh.readline().strip()

    return sc.textFile(params['in_file']) \
        .filter(lambda line: line.strip() != header) \
        .map(lambda line: line.split(',')) \
        .map(lambda record: (record[1], (record[0], record[2]))) \
        .groupByKey() \
        .map(lambda business_set: (business_set[0], set(business_set[1])))


def main():
    # dataset rdd
    dataset_rdd = parse_dataset()
    print(dataset_rdd.take(2))


if __name__ == '__main__':
    # set executables
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # initialize program parameters
    params = parse_args()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(params['app_name']).setMaster("local[*]"))
    sc.setLogLevel('ERROR')

    # run min-hashing + locality sensitive hashing
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
