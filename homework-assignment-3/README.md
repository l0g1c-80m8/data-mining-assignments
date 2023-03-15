# Homework 3: Collaborative Filtering Recommendation Systems

There are two tasks in this homework. The goal is to get familiar with Locality Sensitive
Hashing (LSH), and different types of collaborative-filtering recommendation systems. The dataset
required for this assignment is a subset from the Yelp dataset used in the previous assignments.
For task specifications look at the description file [here](Homework%203%20Description.pdf). 

## Tasks

### Task 1: Jaccard Based LSH
In this task, the Locality Sensitive Hashing algorithm with Jaccard similarity is implemented.
Here a binary system “0 or 1” ratings is used rather than the actual ratings/stars from the users.
Specifically, if a user has rated a business, the user’s contribution in the characteristic matrix is 1. If the
user has not rated the business, the contribution is 0. Businesses whose similarity >= 0.5 are identified.
The signature is generated first using min-hashing followed by LSH by dividing the signature matrix in bands to
maximize precision and recall, while achieving an acceptable time frame for termination.

### Task 2: Recommendation System


## Dataset

The training and validation datasets can be found [here](https://drive.google.com/drive/folders/17JIpck9KcXA2aZYfNGsOFgGTM0qlmPkZ?usp=sharing). The test dataset is hidden and is, thus, inaccessible.

### File 1: ```yelp_train.csv```
This is the subset of the original dataset that is used for training only. <br/>
The file contains records of reviews by users on businesses in the form of stars. A small sample of the dataset is given below.
```
user_id,business_id,stars
vxR_YV0atFxIxfOnF9uHjQ,gTw6PENNGl68ZPUpYWP50A,5.0
o0p-iTC5yTBV5Yab_7es4g,iAuOpYDfOTuzQ6OPpEiGwA,4.0
-qj9ouN0bzMXz1vfEslG-A,5j7BnXXvlS69uLVHrY9Upw,2.0
E43QxgV87Ij6KxMCHcijKw,jUYp798M93Mpcjys_TTgsQ,5.0
T13IBpJITI32a1k41rc-tg,3MntE_HWbNNoyiLGxywjYA,5.0
Q1IENmNc6bdDruACmhy4mg,sPd3E7lFzd_yooiq-ekxtQ,4.0
4bQqil4770ey8GfhBgEGuw,YXohNvMTCmGhFMSQsDZq1g,5.0
0BBUmH7Krcax1RZgbH4fSA,XO2hZb0xC8jTexSHG4SxFg,4.0
QZ_Arlwoj0ghfBvg69rjOw,SUktrYdNQD8k2vvkM4OpfA,5.0
lEw2VL9JCDFk3R5NzahqnA,xVEtGucSRLk5pxxN0t4i6g,4.0
```

### File 2: ```yelp_val.csv```
This is the subset of the original dataset that is used for validation only. <br/>
The file contains records of reviews by users on businesses in the form of stars. A small sample of the dataset is given below.
```
user_id,business_id,stars
wf1GqnKQuvH-V3QN80UOOQ,fThrN4tfupIGetkrz18JOg,5.0
39FT2Ui8KUXwmUt6hnwy-g,uW6UHfONAmm8QttPkbMewQ,5.0
7weuSPSSqYLUFga6IYP4pg,IhNASEZ3XnBHmuuVnWdIwA,4.0
CqaIzLiWaa-lMFYBAsYQxw,G859H6xfAmVLxbzQgipuoA,5.0
yy7shAsNWRbGg-8Y67Dzag,rS39YnrhoXmPqHLzCBjeqw,3.0
Uk1UKBIAwOqhjZdLm3r9zg,5CJL_2-XwCGBmOav4mFdYg,5.0
x-8ZMKKNycT3782Kqf9loA,jgtWfJCJZty_Nctqpdtp3g,5.0
0FVcoJko1kfZCrJRfssfIA,JVK8szNDoy9MNiYSz_MiAA,4.0
LcCRMIDz1JgshpPGYfLDcA,t19vb_4ML2dg5HZ-MF3muA,5.0
C__1BHWTGBNA5s2ZPH289g,h_UvnQfe1cuVICly_kIqHg,2.0
```
