import sys

params = dict()
params['in_file'] = sys.argv[1]
params['out_file'] = sys.argv[2]


with open(params['in_file']) as in_file:
    answer = in_file.read().splitlines(True)[1:]
answer_set = set()
for line in answer:
    row = line.split(',')
    answer_set.add((row[0], row[1]))
with open(params['out_file']) as in_file:
    estimate = in_file.read().splitlines(True)[1:]
estimate_set = set()
for line in estimate:
    row = line.split(',')
    estimate_set.add((row[0], row[1]))
print("Precision:")
print(len(answer_set.intersection(estimate_set)) / len(estimate_set))
print("Recall:")
print(len(answer_set.intersection(estimate_set)) / len(answer_set))
print(answer_set.difference(estimate_set))
