# Homework 1: Analytics on Yelp Data

This homework involves data exploration and analytics on Yelp data set which is available in JSON format. For the exact homework specs refer to the [description file](Homework%201%20Description.pdf).

<!-- TOC -->
* [Homework 1: Analytics on Yelp Data](#homework-1--analytics-on-yelp-data)
  * [Tasks](#tasks)
    * [Task 1: Data Exploration](#task-1--data-exploration)
    * [Task 2: Partition](#task-2--partition)
    * [Task 3: Exploration on Multiple Datasets](#task-3--exploration-on-multiple-datasets)
  * [Dataset](#dataset)
    * [File 1: ```review.json```](#file-1--reviewjson)
    * [File 2: ```business.json```](#file-2--businessjson)
<!-- TOC -->

## Tasks

### Task 1: Data Exploration

- Dataset used: [```review.json```](#file-1-reviewjson) <br/>

The dataset contains the review information from users. The task here is to process the file and
write a program to automatically answer the following questions: <br/>
1. The total number of reviews (0.5 point) <br/>
2. The number of reviews in 2018 (0.5 point) <br/>
3. The number of distinct users who wrote reviews (0.5 point) <br/>
4. The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote (0.5 point) <br/>
5. The number of distinct businesses that have been reviewed (0.5 point) <br/>

Run the program using the following commands (or refer [here](../homework-assignment-0/README.md) for manual execution):
- Locally: ```python python/task1.py dataset/test_review.json out/task1_out.json --local```
- Vocareum: ```./run.sh task1.py ../resource/asnlib/publicdata/review.json ./task1-out.json```

### Task 2: Partition

- Dataset used: [```review.json```](#file-1-reviewjson) <br/>

In this task, a custom partition scheme is compared with the default partition scheme to show
how partitioning of data across the RDD affects the performance of any map-reduce computation.
The trick to design an efficient partition scheme is to study the chain of operations that take
place during the computation and minimizing the implicit shuffle operations between the chained
operations.

Run the program using the following commands (or refer [here](../homework-assignment-0/README.md) for manual execution):
- Locally: ```python python/task2.py dataset/test_review.json out/task2_out.json 4 --local```
- Vocareum: ```./run.sh task2.py ../resource/asnlib/publicdata/review.json ./task2-out.json 21```

### Task 3: Exploration on Multiple Datasets

- Dataset used: [```review.json```](#file-1-reviewjson), [```business.json```](#file-2-businessjson) <br/>

In task 3, two datasets are explored together containing review information (```review.json```) 
and business information (```business.json```). This task requires combining results from both 
datasets to arrive at the final results.

A program is developed to answer the following questions:
1. What are the average stars for each city? (1 point)
2. Compare the execution time of using two methods to print top 10 cities with the highest average stars. (1 point)

Run the program using the following commands (or refer [here](../homework-assignment-0/README.md) for manual execution):
- Locally: ```python python/task3.py dataset/test_review.json dataset/business.json out/task3-A_out.txt out/task3-B_out.json --local```
- Vocareum: ```./run.sh task3.py ../resource/asnlib/publicdata/review.json ../resource/asnlib/publicdata/business.json ./task3-A_out.txt ./task3-B_out.json```

## Dataset

The dataset can be found on these links:
- [Full Dataset](https://www.yelp.com/dataset): This is the full dataset on which the assignment will be graded.
- [Test Dataset](https://drive.google.com/drive/folders/1JlRztnGk5LLD8xYvj6Dp5RgG45YGUNuD?usp=sharing): This is a shorter version of the full dataset which is helpful during the development.

### File 1: ```review.json```

This file is used in [task 1](#task-1-data-exploration), [task 2](#task-2-partition) and in [task 3](#task-3-exploration-on-multiple-datasets). <br/>
This file contains rows of json records as shown below. 
```
{
  "review_id": "Q1sbwvVQXV2734tPgoKj4Q",
  "user_id": "hG7b0MtEbXx5QzbzE6C_VA",
  "business_id": "ujmEBvifdJM6h6RLv4wQIg",
  "stars": 1,
  "useful": 6,
  "funny": 1,
  "cool": 0,
  "text": "Total bill for this horrible service? Over $8Gs. These crooks actually had the nerve to charge us $69 for 3 pills. I checked online the pills can be had for 19 cents EACH! Avoid Hospital ERs at all costs.",
  "date": "2013-05-07 04:34:36"
}
```

### File 2: ```business.json```

This file is used exclusively for [task 3](#task-3-exploration-on-multiple-datasets). <br/>
This file contains rows of json records as shown below. 
```
{
  "business_id": "gnKjwL_1w79qoiV3IC_xQQ",
  "name": "Musashi Japanese Restaurant",
  "address": "10110 Johnston Rd, Ste 15",
  "city": "Charlotte",
  "state": "NC",
  "postal_code": "28210",
  "latitude": 35.092564,
  "longitude": -80.859132,
  "stars": 4,
  "review_count": 170,
  "is_open": 1,
  "attributes": {
    "GoodForKids": "True",
    "NoiseLevel": "u'average'",
    "RestaurantsDelivery": "False",
    "GoodForMeal": "{'dessert': False, 'latenight': False, 'lunch': True, 'dinner': True, 'brunch': False, 'breakfast': False}",
    "Alcohol": "u'beer_and_wine'",
    "Caters": "False",
    "WiFi": "u'no'",
    "RestaurantsTakeOut": "True",
    "BusinessAcceptsCreditCards": "True",
    "Ambience": "{'romantic': False, 'intimate': False, 'touristy': False, 'hipster': False, 'divey': False, 'classy': False, 'trendy': False, 'upscale': False, 'casual': True}",
    "BusinessParking": "{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}",
    "RestaurantsTableService": "True",
    "RestaurantsGoodForGroups": "True",
    "OutdoorSeating": "False",
    "HasTV": "True",
    "BikeParking": "True",
    "RestaurantsReservations": "True",
    "RestaurantsPriceRange2": "2",
    "RestaurantsAttire": "'casual'"
  },
  "categories": "Sushi Bars, Restaurants, Japanese",
  "hours": {
    "Monday": "17:30-21:30",
    "Wednesday": "17:30-21:30",
    "Thursday": "17:30-21:30",
    "Friday": "17:30-22:0",
    "Saturday": "17:30-22:0",
    "Sunday": "17:30-21:0"
  }
}
```
