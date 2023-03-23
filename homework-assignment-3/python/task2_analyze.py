import sys
from collections import defaultdict
from math import sqrt

result_file = sys.argv[1]
validation_file = sys.argv[2]


def read_ratings(filename):
    fh = open(filename, 'r')
    fh.readline()  # read and dismiss header
    return dict(map(
        lambda record: ((record[0], record[1]), float(record[2])),
        list(map(
            lambda line: line.split(','),
            fh.read().split('\n')
        ))[: -1]
    ))


results, validations = read_ratings(result_file), read_ratings(validation_file)
distribution = defaultdict(int)

rmse = 0.0
for key in results.keys():
    diff = abs(results[key] - validations[key])
    if diff < 0:
        distribution['<0'] += 1
    elif diff < 1:
        distribution['<1'] += 1
    elif diff < 2:
        distribution['<2'] += 1
    elif diff < 3:
        distribution['<3'] += 1
    elif diff < 4:
        distribution['<4'] += 1
    elif diff <= 5:
        distribution['<=5'] += 1
    else:
        distribution['>5'] += 1

    rmse += pow(diff, 2)

rmse = sqrt(rmse / sum(distribution.values()))

print(distribution, sum(distribution.values()))
print(rmse)
