from pyspark import SparkContext
import os

os.environ['PYSPARK_PYTHON'] = '/Users/rutvikpatel/.pyenv/versions/3.6.15'
os.environ['PYSPARK_DRIVER_'] = '/Users/rutvikpatel/.pyenv/versions/3.6.15'

sc = SparkContext('local[*]', 'wordCount')

input_file_path = './text.txt'
textRDD = sc.textFile(input_file_path)

word_counts = textRDD.flatMap(lambda line: line.split(' ')) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda count1, count2: count1 + count2) \
    .collect()

for word, count in word_counts:
    print(word, count)
