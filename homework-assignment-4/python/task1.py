"""
Homework Assignment 4
Task 1

Community detection using the Label Propagation algorithm using the GraphFrames library.
https://docs.databricks.com/integrations/graphframes/user-guide-python.html

Construction:
Vertices - All unique users.
Edges - Between users that have a certain minimum number of common businesses.
"""

import os
import sys

from argparse import ArgumentParser
from datetime import datetime
from pyspark import SparkConf, SparkContext


def get_runtime_params():
    # create argument list
    argv = sys.argv + ['hw4-task1']

    # create parser instance
    parser = ArgumentParser(
        prog='Construct Graph',
        description='construct a graph from the dataset based on the specs in the hw document',
        epilog=''.join(['-'] * 60)
    )
    parser.add_argument('executable')
    parser.add_argument('filter_threshold')
    parser.add_argument('in_file')
    parser.add_argument('out_file')
    parser.add_argument('app_name')

    # parse arguments
    runtime_params = parser.parse_args(argv)

    # transform params as required
    runtime_params.filter_threshold = int(runtime_params.filter_threshold)

    # return the params
    return runtime_params


def parse_dataset():
    with open(params.in_file, 'r') as fh:
        header = fh.readline().strip()

        return sc.textFile(params.in_file) \
            .filter(lambda line: line.strip() != header) \
            .map(lambda line: tuple(line.split(',')))


def get_edges_from_dataset(dataset_rdd):
    ub_membership_rdd = dataset_rdd \
        .groupByKey() \
        .map(lambda user_businesses: (user_businesses[0], set(user_businesses[1])))

    return ub_membership_rdd \
        .cartesian(ub_membership_rdd) \
        .filter(lambda pair: pair[0][0] != pair[1][0]) \
        .filter(lambda pair: len(pair[0][1].intersection(pair[1][1])) >= params.filter_threshold) \
        .map(lambda pair: (pair[0][0], pair[1][0])) \
        .collect()


def main():
    # parse the dataset into a rdd
    dataset_rdd = parse_dataset()

    # get vertices - all unique users
    vertices = set(dataset_rdd.collectAsMap().keys())

    # get edges - all users with a certain number of common businesses
    edges = get_edges_from_dataset(dataset_rdd)

    print(len(vertices), len(edges))


if __name__ == '__main__':
    # set executables
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # parse and define params
    params = get_runtime_params()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(params.app_name).setMaster("local[*]"))
    sc.setLogLevel('ERROR')

    # run community detection (based on Label Propagation Algorithm
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)


