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
from itertools import product
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
    parent_count_map = defaultdict(int)
    parent_count_map[root] = 1
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
                        parent_count_map[neighbor] += parent_count_map[node]
                        introduced[neighbor] = level + 1

    return level_tree, parent_count_map


def get_edge_betweenness(parent_count_map, level_tree):
    edge_betweenness = defaultdict(int)
    node_credits = defaultdict(int)
    for level in range(max(level_tree.keys()), -1, -1):
        for node, parents in level_tree[level].items():
            total_credits = 1 + node_credits[node]
            total_paths = min(reduce(
                lambda acc, parent_node: acc + parent_count_map[parent_node],
                parents,
                0
            ), 1)
            for parent in parents:
                edge_credits = total_credits * (parent_count_map[parent] / total_paths)
                edge_betweenness[tuple(sorted([node, parent]))] = edge_credits
                node_credits[parent] += edge_credits

    return edge_betweenness


def girvan_newman(graph_al):
    edge_betweenness = defaultdict(int)
    for node in graph_al:
        level_tree, parent_count_map = get_level_tree(graph_al, node)
        for edge, betweenness in get_edge_betweenness(parent_count_map, level_tree).items():
            edge_betweenness[edge] += betweenness

    for edge in edge_betweenness:
        edge_betweenness[edge] /= 2

    return edge_betweenness


def write_betweenness_to_file(edge_betweenness):
    with open(params.out_betweenness_file, 'w') as fh:
        for edge, betweenness in sorted(edge_betweenness.items(), key=lambda eb: (eb[1], eb[0][0]), reverse=True):
            fh.write('{},{}\n'.format(edge, round(betweenness, params.precision)))


def get_graph_partition(graph_al):
    partitions = []
    visited = set()
    for node in graph_al:
        if node not in visited:
            q = [node]
            visited_in_partition = set()
            while len(q) > 0:
                partition_node = q.pop(0)
                if partition_node not in visited_in_partition:
                    visited_in_partition.add(partition_node)
                    q.extend(graph_al[partition_node] - visited_in_partition)
            partitions.append(visited_in_partition)
            visited.update(visited_in_partition)
    return partitions


def get_modularity_community_division(graph_al, orig_edges, node_degree_map):
    normalizer = sum(node_degree_map.values())
    partition = get_graph_partition(graph_al)
    return partition, reduce(
        lambda modularity, community: reduce(
            lambda acc, node_pair: (
                    acc
                    + (1 if node_pair in orig_edges else 0)
                    - node_degree_map[node_pair[0]] * node_degree_map[node_pair[1]] / normalizer
            ),
            product(community, community),
            modularity
        ),
        partition,
        0
    ) / normalizer


def get_highest_betweenness_edges(edge_betweenness):
    highest_betweenness = max(edge_betweenness.values())
    return list(map(
        lambda edge_betweenness_pair: edge_betweenness_pair[0],
        filter(
            lambda edge_betweenness_pair: edge_betweenness_pair[1] == highest_betweenness,
            edge_betweenness.items()
        )
    ))


def prune_graph(graph_al, curr_edges, edges_to_prune):
    for edge in edges_to_prune:
        graph_al[edge[0]].remove(edge[1])
        graph_al[edge[1]].remove(edge[0])
        curr_edges.remove(edge)
        curr_edges.remove((edge[1], edge[0]))


def get_communities_from_graph(graph_al, orig_edges, node_degree_map):
    communities, modularity = get_modularity_community_division(graph_al, orig_edges, node_degree_map)
    curr_edges = {*orig_edges}

    while len(curr_edges) > 0:
        edge_betweenness = girvan_newman(graph_al)
        highest_betweenness_edges = get_highest_betweenness_edges(edge_betweenness)
        prune_graph(graph_al, curr_edges, highest_betweenness_edges)
        inst_communities, inst_modularity = get_modularity_community_division(graph_al, curr_edges, node_degree_map)
        if inst_modularity > modularity:
            modularity = inst_modularity
            communities = inst_communities

    return communities


def write_communities_to_file(communities):
    communities = sorted(list(map(
        lambda community_node: sorted(map(
            lambda node: "'{}'".format(node),
            community_node
        )),
        communities
    )), key=lambda c: (len(c), c[0]))

    with open(params.out_modularity_file, 'w') as fh:
        for community in communities:
            fh.write('{}\n'.format(', '.join(community)))


def main():
    # get edges - all users with a certain number of common businesses
    edges_rdd = get_edges_from_dataset()

    # create a graph adjacency list
    graph_al = edges_rdd \
        .groupByKey() \
        .mapValues(set) \
        .collectAsMap()

    # get node degrees
    node_degree_map = dict(map(
        lambda node_neighbors_pair: (node_neighbors_pair[0], len(node_neighbors_pair[1])),
        graph_al.items()
    ))

    # task 2 part 1 - get betweenness values for the original graph and save the output to the file
    betweenness = girvan_newman(graph_al)
    write_betweenness_to_file(betweenness)

    # task 2 part 2 - find communities based on highest global modularity measure
    communities = get_communities_from_graph(graph_al, set(edges_rdd.collect()), node_degree_map)
    write_communities_to_file(communities)


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
