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
        out_modularity_file=sys.argv[4],
        precision=5
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


def get_level_tree(graph_al, root):
    level_tree = defaultdict(lambda: defaultdict(set))
    bf_tree = defaultdict(int)
    bf_tree[root] = 1
    level_tree[0] = {root: set()}
    q = [(root, 1)]
    introduced = defaultdict(lambda: float('inf'))
    introduced[root] = 0
    visited = set()
    while len(q) > 0:
        node, level = q.pop(0)
        if node not in visited:
            visited.add(node)
            for neighbor in graph_al[node]:
                if neighbor not in visited:
                    q.append((neighbor, level + 1))
                    if level < introduced[neighbor]:
                        level_tree[level][neighbor].add(node)
                        bf_tree[neighbor] += bf_tree[node]
                        introduced[neighbor] = level + 1

    return level_tree, bf_tree


def get_edge_betweenness(bf_tree, level_tree):
    edge_betweenness = defaultdict(int)
    node_credits = defaultdict(int)
    for level in range(max(level_tree.keys()), -1, -1):
        for node, parents in level_tree[level].items():
            total_credits = 1 + node_credits[node]
            total_paths = reduce(
                lambda acc, parent_node: acc + bf_tree[parent_node],
                parents,
                0
            )
            for parent in parents:
                credits = total_credits * (bf_tree[parent] / total_paths)
                edge_betweenness[tuple(sorted([node, parent]))] = credits
                node_credits[parent] += credits

    return edge_betweenness


def girvan_newman(graph_al):
    edge_betweenness = defaultdict(int)
    for node in graph_al:
        level_tree, bf_tree = get_level_tree(graph_al, node)
        for edge, betweenness in get_edge_betweenness(bf_tree, level_tree).items():
            edge_betweenness[edge] += betweenness

    for edge in edge_betweenness:
        edge_betweenness[edge] /= 2

    return edge_betweenness


def write_betweenness_to_file(edge_betweenness):
    with open(params.out_betweenness_file, 'w') as fh:
        for edge, betweenness in sorted(edge_betweenness.items(), key=lambda c: (c[1], c[0][0]), reverse=True):
            fh.write('{},{}\n'.format(edge, round(betweenness, params.precision)))


def main():
    # get edges - all users with a certain number of common businesses
    edges_rdd = get_edges_from_dataset()

    # task 2 part 1 - get betweenness values for the original graph and save the output to the file
    graph_al = edges_rdd \
        .groupByKey() \
        .mapValues(set) \
        .collectAsMap()
    betweenness = girvan_newman(graph_al)
    write_betweenness_to_file(betweenness)


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