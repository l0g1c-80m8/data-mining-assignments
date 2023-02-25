# Homework 2: Frequent Item-set Mining

This homework is based on frequent item-set mining on a distributed environment. Exact specifications are given in the  [description file](Homework%202%20Description.pdf).

## Tasks

### Task 1: Frequent Item-set mining on simulated dataset

- Dataset used: ```small1.csv```, ```small2.csv```. <br/>

The dataset contains the pairs of user and businesses that they reviewed. The task here is to process the file and
and generate the frequent item-sets for a given support. The data should be processed in two cases. One where users are baskets
for grouping businesses and other where the businesses are baskets for grouping users. <br/> 
The SON algorithm which lends itself to be used along side with other frequent item-set mining techniques like the apriori and the PCY algorithms is used.

Run the program using the following commands (or refer [here](../homework-assignment-0/README.md) for manual execution):
- Locally: ```python python/task1.py <case number (1 | 2)> <support (int)> <input file (small1.csv | small2.csv)> <output file (file path)>```
- Vocareum: ```./run.sh task1.py 1 4 ../resource/asnlib/publicdata/small2.csv ./task1-out.txt```

### Task 2: Frequent Item-set mining on real-world dataset

- Dataset used: ```ta_feng_all_months_merged.csv``` <br/>

In this task, a real dataset is used to perform frequent item-set mining on using the algorithm, written in the previous task.
The data needs to be pre-processed to make it work with the algorithm for frequent item-set mining along with filtering based on a given threshold. <br/>
The algorithm used in the previous task is used again for this task except that the data is first pre-processed and filtered.

Run the program using the following commands (or refer [here](../homework-assignment-0/README.md) for manual execution):
- Locally: ```python python/task2.py <threshold (int)> <support (int)> <input file (ta_feng_all_months_merged.csv)> <output file (file path)>```
- Vocareum: ```./run.sh task2.py 20 50 ../resource/asnlib/publicdata/ta_feng_all_months_merged.csv ./task2-out.txt```


## Dataset

The dataset can be found on these links:
- [Simulated Dataset](https://drive.google.com/drive/folders/1Nqp66TJnE-6aJRBfSJITqta_JZJ7HmE0?usp=sharing): This is the simulated dataset used for task 1.
- [Full Dataset](https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset): This is the real-world data-set used for task 2.

### Simulated Dataset: ```small1.csv```, ```small2.csv```

This file is used in task 2. ```small1.csv``` and ```small2.csv``` are identical except their size.
The former is useful for testing the algorithm while development while the latter is the one used for grading.<br/>
These files contains records of user ids and business ids seperated by a comma in separate lines. A small sample is given below.
```
1,100
1,98
1,101
1,102
2,101
2,99
2,97
2,100
3,98
3,103
3,105
3,102
3,97
3,99
4,102
4,103
4,101
4,99
4,97
```

### Real-world Dataset 2: ```ta_feng_all_months_merged.csv```

This file is used in task 2. This is a real-world data-set. <br/>
These files contains records of transactions made by users at a grocery store over a period of one month. <br/>
 A small sample of the dataset is given below. 
```
"TRANSACTION_DT","CUSTOMER_ID","AGE_GROUP","PIN_CODE","PRODUCT_SUBCLASS","PRODUCT_ID","AMOUNT","ASSET","SALES_PRICE"
"11/1/2000","01104905","45-49","115","110411","4710199010372","2","24","30"
"11/1/2000","00418683","45-49","115","120107","4710857472535","1","48","46"
"11/1/2000","01057331","35-39","115","100407","4710043654103","2","142","166"
"11/1/2000","01849332","45-49","Others","120108","4710126092129","1","32","38"
"11/1/2000","01981995","50-54","115","100205","4710176021445","1","14","18"
"11/1/2000","01741797","35-39","115","110122","0078895770025","1","54","75"
"11/1/2000","00308359","60-64","115","110507","4710192225520","1","85","105"
"11/1/2000","01607000","35-39","221","520503","4712936888817","1","45","68"
"11/1/2000","01057331","35-39","115","320203","4715398106864","2","70","78"
"11/1/2000","00236645","35-39","Unknown","120110","4710126091870","1","43","53"
```
