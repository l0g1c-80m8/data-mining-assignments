# Homework 6: Clustering

In this assignment, the Bradley-Fayyad-Reina (BFR) algorithm is explored for clustering.
The goal is to be familiar with the process of clustering in general on distributed environments and 
various distance measurements. The datasets used here is a synthetic dataset generated to meet
the assumptions of the BFR algorithm.

## Task

The task is to implement Bradley-Fayyad-Reina (BFR) algorithm to cluster the data contained
in ```hw6_clustering.txt```. In BFR, there are three sets of points to be tracked:
```
Discard set (DS), Compression set (CS), Retained set (RS)
```
For each cluster in the DS and CS, the cluster is summarized by:
```
N: The number of points
SUM: the sum of the coordinates of the points
SUMSQ: the sum of squares of coordinates
```

Implementation details of the BFR algorithm: <br/>
Step 1. Load 20% of the data randomly. <br/>
Step 2. Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters)
on the data in memory using the Euclidean distance as the similarity measurement. <br/>
Step 3. In the K-Means result from Step 2, move all the clusters that contain only one point to RS
(outliers). <br/>
Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters. <br/>
Step 5. Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and
generate statistics). <br/>
The initialization of DS has finished, so far, you have K numbers of DS clusters (from Step 5) and some
numbers of RS (from Step 3).
Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input
clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point). <br/>
Step 7. Load another 20% of the data randomly. <br/>
Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign
them to the nearest DS clusters if the distance is < 2 𝑑. <br/>
Step 9. For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and
assign the points to the nearest CS clusters if the distance is < 2 𝑑. <br/>
Step 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS. <br/>
Step 11. Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to
generate CS (clusters with more than one points) and RS (clusters with only one point). <br/>
Step 12. Merge CS clusters that have a Mahalanobis Distance < 2 𝑑. <br/>

Repeat Steps 7 – 12.  If this is the last run (after the last chunk of data), merge CS clusters with DS clusters that
have a Mahalanobis Distance < 2 𝑑. <br/>
At each run, including the initialization step, count and output the number of the discard
points, the number of the clusters in the CS, the number of the compression points, and the number of
the points in the retained set.

## Dataset
Since the BFR algorithm has a strong assumption that the clusters are normally distributed with
independent dimensions, a synthetic dataset is generated by initializing some random centroids and
creating some data points with the centroids and some standard deviations to form the clusters. Outliers are also added 
in the dataset to evaluate the algorithm. Data points which are outliers belong to clusters that are named or indexed as
“-1”. A sample of the dataset is given below.
The first column is the data point index. The second column is the name/index of the cluster that the data point belongs
to. The rest columns represent the features/dimensions of the data point.

The dataset for this task can be found [here](https://drive.google.com/drive/folders/1tLuhdAiVaet4OOYrRwWgdeT-45ZU4WCV?usp=share_link).

```
0,8,-127.64433989643463,-112.93438512156577,-123.4960457961025,114.4630547261514,121.64570029890073,-119.54171797733461,109.9719289517553,134.23436237925256,-117.61527240771153,120.42207629196271
1,4,-38.191305707322314,-36.739481055180704,-34.47221450208468,33.640148757948026,-53.27570482090691,59.21790911677368,53.15109003438039,36.75210113936672,28.951427009179213,41.41404989722435
2,0,194.1751258951049,-214.78572302878496,-199.46759003279055,195.93731866970583,209.634197754483,-192.44259634358372,202.62698763813447,209.16045543699823,197.6554195934683,-202.04341278850256
3,6,-36.018560440437376,40.58411243751584,55.96250080682364,47.5720753795009,-56.61561738372609,-54.944502337157715,-42.84314857713225,-28.76477463042852,-29.123766956654677,-59.3528832139923
4,2,132.2060783793641,-116.45351989798733,123.82220750268765,-115.15470911315373,-126.80354948535924,113.0524942819895,-124.63106833843916,124.77120057287388,-131.35847133488326,-108.9432737700216
5,2,131.07922126519972,-113.91483843467527,120.79689145348577,-110.07370513246919,-128.9562549531342,115.35617093430456,-118.08792142807046,122.73874446358133,-129.9542914778275,-121.51163741617673
6,9,-157.9920320241918,-146.45455755738982,-150.37465359221102,-153.45134572888867,-181.01780539213124,-157.42919227354494,155.31947034832908,-159.10473758897817,174.19025631537878,164.63654248515195
7,7,94.79925670433524,-76.06768897419842,75.83506993034378,88.57595813298732,-99.27444421155536,-79.9109652189898,-83.69054184900928,-71.20927079637288,-81.51143673421231,94.74753126335445
8,2,127.55831413687531,-114.30544099475115,116.61252996772919,-107.79996293830662,-122.79695366181446,110.33816651449428,-120.52550906786149,125.6085587319683,-125.71473782011275,-121.29386630466168
9,2,131.38976292127936,-108.33389631754892,125.44486660770555,-109.22084758510982,-121.42566821031811,116.43393718828446,-125.4469519283723,125.63834083134527,-123.65190775403593,-114.11098567917787
```