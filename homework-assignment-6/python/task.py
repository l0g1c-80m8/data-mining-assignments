"""
Homework Assignment 6
Task 1

Implement the Bradley-Fayyad-Reina (BFR) Algorithm to cluster multidimensional data
"""

import sys
import os

from argparse import Namespace
from datetime import datetime
from functools import reduce
from pyspark import SparkConf, SparkContext
from math import ceil
from random import shuffle
from sklearn.cluster import KMeans as kmc


def get_runtime_params():
    if len(sys.argv) < 4:
        print('ERR: Expected three arguments: (i/p file, n_clusters, o/p file).')
        exit(1)

    # return the params
    return Namespace(
        APP_NAME='hw6-task',
        IN_FILE=sys.argv[1],
        N_CLUSTERS=int(sys.argv[2]),
        OUT_FILE=sys.argv[3],
        CHUNK_SIZE_PERCENT=20  # 20 %
    )


def get_data_chunks():
    with open(PARAMS.IN_FILE, 'r') as fh:
        header = fh.readline().strip()

        data = sc.textFile(PARAMS.IN_FILE) \
            .filter(lambda line: line.strip() != header) \
            .map(lambda line: line.split(',')) \
            .map(lambda record: (record[0], tuple(record[2:]))) \
            .collect()
        shuffle(data)
        n_chunks = ceil(100 / PARAMS.CHUNK_SIZE_PERCENT)
        chunk_size = ceil(len(data) / n_chunks)

        return reduce(
            lambda chunks, chunk_num: chunks + [data[chunk_num * chunk_size:(chunk_num + 1) * chunk_size]],
            range(n_chunks),
            list()
        )


def main():
    # chunked data
    data_chunks = get_data_chunks()
    print(list(map(len, data_chunks)))


if __name__ == '__main__':
    # set executables
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # get runtime params
    PARAMS = get_runtime_params()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(PARAMS.APP_NAME).setMaster("local[*]"))
    sc.setLogLevel('ERROR')

    # run clustering
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
