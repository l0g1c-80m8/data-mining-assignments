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
from itertools import product
from pyspark import SparkConf, SparkContext
from math import ceil, sqrt
from random import shuffle
from sklearn.cluster import KMeans
from uuid import uuid4


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


class Cluster:
    clusters = None
    feature_length = None

    def __init__(self, feature_length):
        self.feature_length = feature_length
        self.clusters = defaultdict(lambda: (
            0,
            tuple([0] * self.feature_length),
            tuple([0] * self.feature_length),
            list()
        ))

    def insert(self, data_point, label):
        curr_cluster = self.clusters[label]
        self.clusters[label] = (
            curr_cluster[0] + 1,
            tuple([data_point[i] + curr_cluster[1][i] for i in range(self.feature_length)]),
            tuple([data_point[i] ** 2 + curr_cluster[2][i] for i in range(self.feature_length)]),
            curr_cluster[3] + [data_point]
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
        data_points = sc.textFile(PARAMS.IN_FILE) \
            .map(lambda line: line.split(',')) \
            .map(lambda line: list(map(float, line)))

        data_point_map = data_points \
            .map(lambda record: (tuple(record[2:]), int(record[0]))) \
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
        ds.insert_all(cluster_data, str(uuid4()))


def move_to_cs_rs(km_inst, cs, rs, data_chunks, curr_chunk):
    cluster_groups = get_cluster_group(km_inst, data_chunks, curr_chunk)

    for cluster_label, cluster_data in cluster_groups.items():
        if len(cluster_data) > 1:
            cs.insert_all(cluster_data, str(uuid4()))
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


def get_mahalanobis_dist_point(cluster_stats, point):
    dist = 0
    for idx, dim in enumerate(point):
        mean = cluster_stats[1][idx] / cluster_stats[0]
        variance = cluster_stats[2][idx] / cluster_stats[0] - mean
        dist += pow(dim - mean, 2) / variance

    return sqrt(dist)


def get_mahalanobis_dist_clusters(cluster_1, cluster_2):
    dist_1, dist_2 = 0, 0
    for idx in range(len(cluster_1[1])):
        mean_1, mean_2 = cluster_1[1][idx] / cluster_1[0], cluster_2[1][idx] / cluster_2[0]
        var_1, var_2 = cluster_1[2][idx] / cluster_1[0] - mean_1, cluster_2[2][idx] / cluster_2[0] - mean_2
        dist_1 += pow(mean_1 - mean_2, 2) / var_1
        dist_2 += pow(mean_1 - mean_2, 2) / var_2

    return (sqrt(dist_1) + sqrt(dist_2)) / 2


def assign_to_cluster(point, cluster_set):
    min_label, min_dist = -1, float('inf')
    for label, cluster in cluster_set.clusters.items():
        dist = get_mahalanobis_dist_point(cluster, point)
        if dist < min_dist:
            min_label = label
            min_dist = dist
    if min_dist < 2 * cluster_set.feature_length ** 0.5:
        cluster_set.insert(point, min_label)
        return True
    return False


def assign_points(data_chunk, curr_chunk, ds, cs, rs):
    for point in data_chunk[curr_chunk]:
        assigned = assign_to_cluster(point, ds)
        if assigned:
            continue
        assigned = assign_to_cluster(point, cs)
        if assigned:
            continue
        rs.insert(point)


def merge_cs_clusters(cs):
    while True:
        min_dist = float('inf')
        min_pair = (-1, -1)
        for cluster_1, cluster_2 in product(cs.clusters.items(), cs.clusters.items()):
            if cluster_1[0] == cluster_2[0] or cluster_1[0] not in cs.clusters or cluster_2[0] not in cs.clusters:
                continue
            dist = get_mahalanobis_dist_clusters(cluster_1[1], cluster_2[1])
            if min_dist > dist:
                min_dist = dist
                min_pair = (cluster_1, cluster_2)
        if min_dist > 2 * sqrt(cs.feature_length):
            break
        cluster_1, cluster_2 = min_pair
        cs.clusters[cluster_1[0]] = (
            cluster_1[1][0] + cluster_2[1][0],
            tuple([cluster_1[1][1][idx] + cluster_2[1][1][idx] for idx in range(cs.feature_length)]),
            tuple([cluster_1[1][2][idx] + cluster_2[1][2][idx] for idx in range(cs.feature_length)]),
            cluster_1[1][3] + cluster_2[1][3]
        )
        cs.clusters.pop(cluster_2[0])


def merge_cs_ds_clusters(cs, ds):
    while True:
        min_dist = float('inf')
        min_pair = (-1, -1)
        for cluster_1, cluster_2 in product(ds.clusters.items(), cs.clusters.items()):
            if cluster_1[0] == cluster_2[0] or cluster_1[0] not in ds.clusters or cluster_2[0] not in cs.clusters:
                continue
            dist = get_mahalanobis_dist_clusters(cluster_1[1], cluster_2[1])
            if min_dist > dist:
                min_dist = dist
                min_pair = (cluster_1, cluster_2)
        if min_dist > 2 * sqrt(ds.feature_length):
            break
        cluster_1, cluster_2 = min_pair
        ds.clusters[cluster_1[0]] = (
            cluster_1[1][0] + cluster_2[1][0],
            tuple([cluster_1[1][1][idx] + cluster_2[1][1][idx] for idx in range(cs.feature_length)]),
            tuple([cluster_1[1][2][idx] + cluster_2[1][2][idx] for idx in range(cs.feature_length)]),
            cluster_1[1][3] + cluster_2[1][3]
        )
        cs.clusters.pop(cluster_2[0])


def get_clustering_labels(data_point_map, ds, cs, rs):
    label_map = dict()
    cluster_labels_map = dict()
    for idx, label in enumerate(list(ds.clusters.keys()) + list(cs.clusters.keys())):
        cluster_labels_map[label] = idx

    for cluster_label, cluster in ds.clusters.items():
        for data_point in cluster[3]:
            label_map[data_point_map[tuple(data_point)]] = cluster_labels_map[cluster_label]

    for cluster_label, cluster in cs.clusters.items():
        for data_point in cluster[3]:
            label_map[data_point_map[tuple(data_point)]] = cluster_labels_map[cluster_label]

    for data_point in rs.data_points:
        label_map[data_point_map[tuple(data_point)]] = -1

    return label_map


def write_output_to_file(ir, labels):
    with open(PARAMS.OUT_FILE, 'w') as fh:
        fh.write('The intermediate results:\n')
        for result in ir:
            fh.write('Round {}: {}, {}, {}, {}\n'.format(*result))

        fh.write('\nThe clustering results:\n')
        for label in sorted(labels.items(), key=lambda l: l[0]):
            fh.write('{},{}\n'.format(label[0], label[1]))


def main():
    # chunked data
    data_chunks, data_point_map = get_data_chunks()

    # init control vars and runtime vars
    rs = RS()
    cs = Cluster(len(data_chunks[0][0]))
    ds = Cluster(len(data_chunks[0][0]))
    curr_chunk = 0
    intermediate_results = list()

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

    for curr_chunk in range(1, len(data_chunks)):
        intermediate_results.append((
            curr_chunk,
            sum(map(lambda c: c[0], ds.clusters.values())),
            len(cs.clusters),
            sum(map(lambda c: c[0], cs.clusters.values())),
            len(rs.data_points)
        ))
        assign_points(data_chunks, curr_chunk, ds, cs, rs)
        km_inst = get_km_inst(min(PARAMS.N_CLUSTERS * PARAMS.SCALE_SMALL, len(rs.data_points)))
        data_points = list(rs.pop_all_points())
        km_inst.fit(data_points)
        move_to_cs_rs(km_inst, cs, rs, [data_points], 0)
        merge_cs_clusters(cs)

    merge_cs_ds_clusters(cs, ds)

    intermediate_results.append((
        curr_chunk + 1,
        sum(map(lambda c: c[0], ds.clusters.values())),
        len(cs.clusters),
        sum(map(lambda c: c[0], cs.clusters.values())),
        len(rs.data_points)
    ))

    cluster_labels = get_clustering_labels(data_point_map, ds, cs, rs)
    write_output_to_file(intermediate_results, cluster_labels)


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
