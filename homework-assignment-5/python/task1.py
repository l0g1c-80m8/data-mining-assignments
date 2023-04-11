"""
Homework Assignment 5
Task 1

Implement a bloom filter on a simulated datastream.
"""

import os
import sys

from argparse import Namespace
from datetime import datetime
from pyspark import SparkConf, SparkContext


def get_runtime_params():
    if len(sys.argv) < 5:
        print('ERR: Expected four arguments: (i/p file, stream_size, num_of_asks, o/p file).')
        exit(1)

    # return the params
    return Namespace(
        APP_NAME='hw5-task1',
        INPUT_FILE=sys.argv[1],
        STREAM_SIZE=sys.argv[2],
        NUM_OF_ASKS=sys.argv[3],
        Output_FILE=sys.argv[4],
    )


def main():
    pass


if __name__ == '__main__':
    # set executables
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # get runtime params
    PARAMS = get_runtime_params()

    # create spark context
    sc = SparkContext(conf=SparkConf().setAppName(PARAMS.APP_NAME).setMaster("local[*]"))
    sc.setLogLevel('ERROR')

    # run community detection (based on Label Propagation Algorithm
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
