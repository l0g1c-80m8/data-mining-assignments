# Homework 4: Community Detection

In this assignment, the Girvan-Newman algorithm is implemented along with the exploration of the spark GraphFrames
library. The ```ub_sample_data.csv``` dataset is used to find users who have similar business tastes.
The goal of this assignment is to understand how to use the Girvan-Newman algorithm to detect communities 
in an efficient  way within a distributed environment.

<!-- TOC -->
* [Homework 4: Community Detection](#homework-4-community-detection)
  * [Tasks](#tasks)
    * [Task 1: Community Detection using the GraphFrames library (Label Propagation Algorithm)](#task-1-community-detection-using-the-graphframes-library-label-propagation-algorithm)
    * [Task 2: Community Detection using the Girvan-Newman algorithm (maximizing modularity)](#task-2-community-detection-using-the-girvan-newman-algorithm-maximizing-modularity)
      * [Task 2.1. Betweenness Calculation](#task-21-betweenness-calculation)
      * [Task 2.2. Community Detection](#task-22-community-detection)
  * [Dataset](#dataset)
<!-- TOC -->

## Tasks

### Task 1: Community Detection using the GraphFrames library (Label Propagation Algorithm)

To construct the social network graph, it is assumed that each node is uniquely labeled and that links are
undirected and unweighted. Each node represents a user. There should be an edge between two nodes if the number of
common  businesses reviewed by two users is greater than or equivalent to the filter threshold (runtime parameter).

If the user node has no edge, that node is not included in the graph.

The Spark GraphFrames library is used for this task to detect communities in the network graph constructed from the
dataset. The library provides the implementation of the Label Propagation Algorithm
(LPA) which was proposed by Raghavan, Albert, and Kumara in 2007. It is an iterative community
detection solution whereby information “flows” through the graph based on underlying edge structure.

References for GraphFrames:
 - [Python](https://docs.databricks.com/spark/latest/graph-analysis/graphframes/user-guide-python.html)
 - [Scala](https://docs.databricks.com/spark/latest/graph-analysis/graphframes/user-guide-scala.html)


### Task 2: Community Detection using the Girvan-Newman algorithm (maximizing modularity)

In this task, the Girvan-Newman algorithm is implemented to detect the communities in the
network graph. The graph construction is the same here as it is for Task 1.

#### Task 2.1. Betweenness Calculation
In this part, the betweenness of each edge in the original graph is calculated using the Girvan-Newman algorithm.

#### Task 2.2. Community Detection
In this part, the communities are detected using the algorithm implemented for part 1.
The edges with the highest betweenness centrality measure are removed and the betweenness in the remaining graph is
updated. This process is done recursively until all edges have been removed. Meanwhile, the modularity of the graph is
also tracked and the partition yielding the highest modularity is decidedly declared as the set of communities in the
graph.


## Dataset

The dataset for this task can be found [here](https://drive.google.com/drive/folders/1wJso0NNgK9jv4fjfRTSepYl58s1LPZQs?usp=share_link).

This dataset has a single file ```ub_sample_data.csv```.
It contains user and business pairs in separate lines. Each line represents the membership of a user to a business.

The first few lines of the files are given below:
```
user_id,business_id
39FT2Ui8KUXwmUt6hnwy-g,RJSFI7mxGnkIIKiJCufLkg
39FT2Ui8KUXwmUt6hnwy-g,fThrN4tfupIGetkrz18JOg
39FT2Ui8KUXwmUt6hnwy-g,mvLdgkwBzqllHWHwS-ZZgQ
39FT2Ui8KUXwmUt6hnwy-g,uW6UHfONAmm8QttPkbMewQ
39FT2Ui8KUXwmUt6hnwy-g,T70pMoTP008qYLsIvFCXdQ
39FT2Ui8KUXwmUt6hnwy-g,dJS3iH-odljqWS9MKEFsBA
39FT2Ui8KUXwmUt6hnwy-g,G859H6xfAmVLxbzQgipuoA
39FT2Ui8KUXwmUt6hnwy-g,5CJL_2-XwCGBmOav4mFdYg
39FT2Ui8KUXwmUt6hnwy-g,0ptR21GHRuQ1MFtxGNcxzw
39FT2Ui8KUXwmUt6hnwy-g,JVK8szNDoy9MNiYSz_MiAA
39FT2Ui8KUXwmUt6hnwy-g,T-TES2u1IA2THb8uBhNdCA
39FT2Ui8KUXwmUt6hnwy-g,n_-CwlqV8fzEIUPky8Ibtw
39FT2Ui8KUXwmUt6hnwy-g,QjZFYd5hme7EHegpuJngMQ
39FT2Ui8KUXwmUt6hnwy-g,9p-cpmHaga-EXyc6ZzYCcQ
```