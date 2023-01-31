# Homework 1: Analytics on Yelp Data

This homework involves data exploration and analytics on Yelp data set which is available in JSON format.

## Tasks

### Task 1: Data Exploration

- Dataset used: [```test_review.json```](#file-1--testreviewjson) <br/>

The dataset contains the review information from users. The task here is to process the file and
write a program to automatically answer the following questions: <br/>
1. The total number of reviews (0.5 point) <br/>
2. The number of reviews in 2018 (0.5 point) <br/>
3. The number of distinct users who wrote reviews (0.5 point) <br/>
4. The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote (0.5 point) <br/>
5. The number of distinct businesses that have been reviewed (0.5 point) <br/>

## Dataset

The dataset can be found on these links:
- [Full Dataset](https://www.yelp.com/dataset): This is the full dataset on which the assignment will be graded.
- [Test Dataset](https://drive.google.com/drive/folders/1JlRztnGk5LLD8xYvj6Dp5RgG45YGUNuD?usp=sharing): This is a shorter version of the full dataset which is helpful during the development.

### File 1: ```test_review.json```

This file is used in [task 1](#task-1--data-exploration) and in [task 3](#task-3--data-exploration). <br/>
This file contains rows of json records as shown below. 
```
{"review_id":"Q1sbwvVQXV2734tPgoKj4Q","user_id":"hG7b0MtEbXx5QzbzE6C_VA","business_id":"ujmEBvifdJM6h6RLv4wQIg","stars":1.0,"useful":6,"funny":1,"cool":0,"text":"Total bill for this horrible service? Over $8Gs. These crooks actually had the nerve to charge us $69 for 3 pills. I checked online the pills can be had for 19 cents EACH! Avoid Hospital ERs at all costs.","date":"2013-05-07 04:34:36"}
```
