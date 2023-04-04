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

from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from functools import reduce
from graphframes import GraphFrame
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SparkSession


def get_runtime_params():
    if len(sys.argv) < 4:
        # expected arguments: script path, dataset path, output file path
        print('ERR: Expected three arguments: (threshold, input file, output file).')
        exit(1)

    # return the params
    return Namespace(
        app_name='hw4-task1',
        filter_threshold=int(sys.argv[1]),
        in_file=sys.argv[2],
        out_file=sys.argv[3]
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


def write_results_to_file(communities):
    def _add_node_to_community(acc, row):
        acc[row.label].append(row.id)
        return acc

    community_map = dict(map(
        lambda label_value_pair: (label_value_pair[0], sorted(label_value_pair[1])),
        reduce(_add_node_to_community, communities, defaultdict(list)).items()
    ))

    with open(params.out_file, 'w') as fh:
        for community in sorted(community_map.values(), key=lambda c: (len(c), c[0])):
            fh.write('{}\n'.format((', '.join(map(lambda user_id: "'{}'".format(user_id), community)))))


def main():
    # get edges - all users with a certain number of common businesses
    edges = get_edges_from_dataset()

    # get vertices - all unique users
    vertices_row = Row("id")
    vertices = edges \
        .map(lambda user_pair: user_pair[0]) \
        .distinct() \
        .map(vertices_row)

    # create graph frame
    vertices_df = vertices.toDF(["id"])
    edges_df = edges.toDF(["src", "dst"])
    gf = GraphFrame(vertices_df, edges_df)

    # run label propagation algorithm
    communities = gf.labelPropagation(maxIter=5).collect()

    # collect result and write to file
    write_results_to_file(communities)


if __name__ == '__main__':
    # set executables
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"

    # parse and define params
    params = get_runtime_params()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(params.app_name).setMaster("local[*]"))
    sc.setLogLevel('ERROR')
    spark = SparkSession(sc)

    # run community detection (based on Label Propagation Algorithm
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
