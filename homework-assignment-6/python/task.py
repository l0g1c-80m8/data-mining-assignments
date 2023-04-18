"""
Homework Assignment 6
Task 1

Implement the Bradley-Fayyad-Reina (BFR) Algorithm to cluster multi-dimensional data
"""

import sys

from argparse import Namespace
from datetime import datetime


def get_runtime_params():
    if len(sys.argv) < 4:
        print('ERR: Expected three arguments: (i/p file, n_clusters, o/p file).')
        exit(1)

    # return the params
    return Namespace(
        APP_NAME='hw6-task',
        INPUT_FILE=sys.argv[1],
        N_CLUSTERS=int(sys.argv[2]),
        OUTPUT_FILE=sys.argv[3]
    )


def main():
    pass


if __name__ == '__main__':
    # get runtime params
    PARAMS = get_runtime_params()
    print(PARAMS)

    # run community detection (based on Label Propagation Algorithm
    start_ts = datetime.now()
    main()
    end_ts = datetime.now()
    print('Duration: ', (end_ts - start_ts).total_seconds())

    # exit without errors
    exit(0)
