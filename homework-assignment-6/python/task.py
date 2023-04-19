"""
Homework Assignment 6
Task 1

Implement the Bradley-Fayyad-Reina (BFR) Algorithm to cluster multidimensional data
"""

import sys
import os

from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from functools import reduce
from pyspark import SparkConf, SparkContext
from math import ceil
from random import shuffle
from sklearn.cluster import KMeans


class RS:
    data_points = None

    def __init__(self):
        self.data_points = set()

    def __str__(self):
        return str(self.data_points)

    def insert(self, data_point):
        self.data_points.add(tuple(data_point))

    def insert_all(self, data_points):
        for data_point in data_points:
            self.insert(data_point)


class DS:
    clusters = None

    def __init__(self, feature_length):
        self.clusters = defaultdict(lambda: (
            0,
            tuple([0] * feature_length),
            tuple([0] * feature_length)
        ))

    def insert(self, data_point, label):
        curr_cluster = self.clusters[label]
        self.clusters[label] = (
            curr_cluster[0] + 1,
            tuple([data_point[i] + curr_cluster[1][i] for i in range(len(data_point))]),
            tuple([data_point[i] ** 2 + curr_cluster[2][i] for i in range(len(data_point))])
        )

    def insert_all(self, data_points, label):
        for data_point in data_points:
            self.insert(data_point, label)


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
        N_CLUSTERS_SCALE=5
    )


def get_data_chunks():
    with open(PARAMS.IN_FILE, 'r') as fh:
        header = fh.readline().strip()

        data_points = sc.textFile(PARAMS.IN_FILE) \
            .filter(lambda line: line.strip() != header) \
            .map(lambda line: line.split(',')) \
            .map(lambda line: list(map(float, line)))

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


def move_to_rs(km_inst, rs, data_chunks, curr_chunk):
    cluster_groups = defaultdict(list)
    for idx, label in enumerate(km_inst.labels_):
        cluster_groups[label.item()].append(data_chunks[curr_chunk][idx])

    for cluster_data in cluster_groups.values():
        if len(cluster_data) == 1:
            rs.insert_all(cluster_data)

    data_chunks[curr_chunk] = list(filter(
        lambda data_point: tuple(data_point) not in rs.data_points,
        data_chunks[curr_chunk]
    ))


def move_to_ds(km_inst, ds, data_chunks, curr_chunk):
    cluster_groups = defaultdict(list)
    for idx, label in enumerate(km_inst.labels_):
        cluster_groups[label.item()].append(data_chunks[curr_chunk][idx])

    for cluster_label, cluster_data in cluster_groups.items():
        ds.insert_all(cluster_data, cluster_label)


def main():
    # create instances to run kmeans clustering
    km_inst_loose = KMeans(
        n_clusters=PARAMS.N_CLUSTERS * PARAMS.N_CLUSTERS_SCALE,
        init='k-means++',
        n_init='auto',
        max_iter=PARAMS.KM_MAX_ITERS,
        tol=PARAMS.KM_TOL
    )

    km_inst_tight = KMeans(
        n_clusters=PARAMS.N_CLUSTERS,
        init='k-means++',
        n_init='auto',
        max_iter=PARAMS.KM_MAX_ITERS,
        tol=PARAMS.KM_TOL
    )

    # chunked data
    data_chunks, data_point_map = get_data_chunks()

    # init control vars and runtime vars
    rs = RS()
    ds = DS(len(data_chunks[0][0]))
    curr_chunk = 0

    # run km_inst_loose for generating RS
    km_inst_loose.fit(data_chunks[curr_chunk])
    move_to_rs(km_inst_loose, rs, data_chunks, curr_chunk)
    print(len(rs.data_points))
    km_inst_tight.fit(data_chunks[curr_chunk])
    move_to_ds(km_inst_tight, ds, data_chunks, curr_chunk)


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
