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

    def pop_all_points(self):
        data_points = self.data_points.copy()
        self.data_points = set()
        return data_points


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
        SCALE_LARGE=50,
        SCALE_SMALL=2
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


def get_cluster_group(km_inst, data_chunks, curr_chunk):
    cluster_groups = defaultdict(list)
    for idx, label in enumerate(km_inst.labels_):
        cluster_groups[label.item()].append(data_chunks[curr_chunk][idx])
    return cluster_groups


def move_to_rs(km_inst, rs, data_chunks, curr_chunk):
    cluster_groups = get_cluster_group(km_inst, data_chunks, curr_chunk)

    for cluster_data in cluster_groups.values():
        if len(cluster_data) == 1:
            rs.insert_all(cluster_data)

    data_chunks[curr_chunk] = list(filter(
        lambda data_point: tuple(data_point) not in rs.data_points,
        data_chunks[curr_chunk]
    ))


def move_to_ds(km_inst, ds, data_chunks, curr_chunk):
    cluster_groups = get_cluster_group(km_inst, data_chunks, curr_chunk)

    for cluster_label, cluster_data in cluster_groups.items():
        ds.insert_all(cluster_data, cluster_label)


def move_to_cs_rs(km_inst, cs, rs, data_chunks, curr_chunk):
    cluster_groups = get_cluster_group(km_inst, data_chunks, curr_chunk)

    for cluster_label, cluster_data in cluster_groups.items():
        if len(cluster_data) > 1:
            cs.insert_all(cluster_data, cluster_label)
        else:
            rs.insert_all(cluster_data)


def get_km_inst(n_clusters):
    return KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init='auto',
        max_iter=PARAMS.KM_MAX_ITERS,
        tol=PARAMS.KM_TOL
    )


def main():
    # chunked data
    data_chunks, data_point_map = get_data_chunks()

    # init control vars and runtime vars
    rs = RS()
    cs = DS(len(data_chunks[0][0]))
    ds = DS(len(data_chunks[0][0]))
    curr_chunk = 0

    # run k means for generating RS
    km_inst = get_km_inst(PARAMS.N_CLUSTERS * PARAMS.SCALE_LARGE)
    km_inst.fit(data_chunks[curr_chunk])
    move_to_rs(km_inst, rs, data_chunks, curr_chunk)

    # run k means for generating DS
    km_inst = get_km_inst(PARAMS.N_CLUSTERS)
    km_inst.fit(data_chunks[curr_chunk])
    move_to_ds(km_inst, ds, data_chunks, curr_chunk)

    # run k means for generating CS and RS
    km_inst = get_km_inst(min(PARAMS.N_CLUSTERS * PARAMS.SCALE_SMALL, len(rs.data_points)))
    data_points = list(rs.pop_all_points())
    km_inst.fit(data_points)
    move_to_cs_rs(km_inst, cs, rs, [data_points], 0)

    print(sum(map(lambda c: c[0], cs.clusters.values())))
    print(sum(map(lambda c: c[0], ds.clusters.values())))
    print(len(rs.data_points))
    print(len(data_chunks[1]))


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
