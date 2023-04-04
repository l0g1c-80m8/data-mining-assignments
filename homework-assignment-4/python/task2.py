"""
Homework Assignment 4
Task 2

Community detection using the Girvan-Newman algorithm.

Construction:
Vertices - All unique users.
Edges - Between users that have a certain minimum number of common businesses.
"""

import os
import sys

from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from functools import reduce
from pyspark import SparkConf, SparkContext


def get_runtime_params():
    if len(sys.argv) < 5:
        # expected arguments: script path, dataset path, output file path
        print('ERR: Expected four arguments: (threshold, input file, output file (betweenness), output file ('
              'modularity)).')
        exit(1)

    # return the params
    return Namespace(
        app_name='hw4-task2',
        filter_threshold=int(sys.argv[1]),
        in_file=sys.argv[2],
        out_betweenness_file=sys.argv[3],
        out_modularity_file=sys.argv[4]
    )


def get_edges_from_dataset():
    with open(params.in_file, 'r') as fh:
        header = fh.readline().strip()

        ub_membership_rdd = sc.textFile(params.in_file) \
            .filter(lambda line: line.strip() != header) \
            .map(lambda line: tuple(line.split(','))) \
            .groupByKey() \
            .map(lambda user_businesses: (user_businesses[0], set(user_businesses[1])))

        return ub_membership_rdd \
            .cartesian(ub_membership_rdd) \
            .filter(lambda pair: pair[0][0] != pair[1][0]) \
            .filter(lambda pair: len(pair[0][1].intersection(pair[1][1])) >= params.filter_threshold) \
            .map(lambda pair: (pair[0][0], pair[1][0]))


def main():
    # get edges - all users with a certain number of common businesses
    all_edges = get_edges_from_dataset()
    edges = all_edges \
        .map(lambda pair: tuple(sorted(pair))) \
        .distinct()

    # get vertices - all unique users
    vertices = all_edges \
        .map(lambda user_pair: user_pair[0]) \
        .distinct()

    print(edges.collect())
    print(vertices.collect())


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
