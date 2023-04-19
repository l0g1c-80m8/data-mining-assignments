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
        CHUNK_SIZE_PERCENT=20,  # 20 %
        KM_MAX_ITERS=300,
        KM_TOL=1e-04,
        N_CLUSTERS_SCALE=3
    )


def get_data_chunks():
    with open(PARAMS.IN_FILE, 'r') as fh:
        header = fh.readline().strip()

        data_points = sc.textFile(PARAMS.IN_FILE) \
            .filter(lambda line: line.strip() != header) \
            .map(lambda line: line.split(','))

        data_point_map = data_points \
            .map(lambda record: (record[0], record[2:])) \
            .collectAsMap()

        data = data_points \
            .map(lambda record: record[2:]) \
            .collect()

        shuffle(data)
        n_chunks = ceil(100 / PARAMS.CHUNK_SIZE_PERCENT)
        chunk_size = ceil(len(data) / n_chunks)

        return reduce(
            lambda chunks, chunk_num: chunks + [data[chunk_num * chunk_size:(chunk_num + 1) * chunk_size]],
            range(n_chunks),
            list()
        ), data_point_map


def main():
    # create instances to run kmeans clustering
    km_inst_loose = kmc(
        n_clusters=PARAMS.N_CLUSTERS * PARAMS.N_CLUSTERS_SCALE,
        init='random',
        n_init='auto',
        max_iter=PARAMS.KM_MAX_ITERS,
        tol=PARAMS.KM_TOL
    )

    km_inst_tight = kmc(
        n_clusters=PARAMS.N_CLUSTERS,
        init='random',
        n_init='auto',
        max_iter=PARAMS.KM_MAX_ITERS,
        tol=PARAMS.KM_TOL
    )

    # chunked data
    data_chunks, data_point_map = get_data_chunks()

    # init control vars
    curr_chunk = 0

    # run km_inst_loose for generating RS
    km_inst_loose.fit(data_chunks[curr_chunk])
    print(km_inst_loose.n_iter_)


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
