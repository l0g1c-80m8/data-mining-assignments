# Competition Project (Recommendation System)

The performance of the recommendation system from [homework 3 task 2](../homework-assignment-3) is 
improved to get better accuracy and efficiency. Exact details are available in the [project specifications](Competition%20Project%20Description.pdf). 


## Task

Here various features are used form the ```user.json```, ```business.json```, ```checkin.json```, ```tip.json```, ```photo.json``` and ```review_train.json``` files in addition to the training
dataset to train a model based on a regressor based on Decision Tree using the [```XGBRegressor```](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
library. For the final submission, even the validation dataset is used to train the model for the test set which remains hidden.

#### Warning: Spaghetti 🍝 Code in this Assignment!

## Dataset

The training and validation datasets can be found [here](https://drive.google.com/drive/folders/17JIpck9KcXA2aZYfNGsOFgGTM0qlmPkZ?usp=sharing). The test dataset is hidden and is, thus, inaccessible.
Note: This dataset is the same as homework 3 dataset. Additional files are used to improve the performance of the recommendation system from 
by acute feature engineering on the available datasets.

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

### File 3: ```user.json```
This file is used to extract user features for the model-based recommendation system implementation
used for tasks 2.2 and 2.3. This file contains records of users with some attributes as shown below.
<br/> Note: There may be other attributes not present in this record.
```
{
  "user_id": "s4FoIXE_LSGviTHBe8dmcg",
  "name": "Shashank",
  "review_count": 3,
  "yelping_since": "2017-06-18",
  "friends": "None",
  "useful": 0,
  "funny": 0,
  "cool": 0,
  "fans": 0,
  "elite": "None",
  "average_stars": 3,
  "compliment_hot": 0,
  "compliment_more": 0,
  "compliment_profile": 0,
  "compliment_cute": 0,
  "compliment_list": 0,
  "compliment_note": 0,
  "compliment_plain": 0,
  "compliment_cool": 0,
  "compliment_funny": 0,
  "compliment_writer": 0,
  "compliment_photos": 0
}
```

### File 4: ```business.json```
This file is used to extract business features for the model-based recommendation system implementation
used for tasks 2.2 and 2.3. This file contains records of businesses with some attributes as shown below.
<br/> Note: There may be other attributes not present in this record.
```
{
  "business_id": "cuXCQM-9VwpZlSneEY1b3w",
  "name": "Indian Street Food Company",
  "neighborhood": "Mount Pleasant and Davisville",
  "address": "1701 Bayview Avenue",
  "city": "Toronto",
  "state": "ON",
  "postal_code": "M4G 3C1",
  "latitude": 43.708002,
  "longitude": -79.375814,
  "stars": 3.5,
  "review_count": 51,
  "is_open": 1,
  "attributes": {
    "Alcohol": "full_bar",
    "Ambience": "{'romantic': False, 'intimate': False, 'classy': False, 'hipster': False, 'touristy': False, 'trendy': True, 'upscale': False, 'casual': False}",
    "BikeParking": "True",
    "BusinessParking": "{'garage': False, 'street': True, 'validated': False, 'lot': False, 'valet': False}",
    "Caters": "True",
    "CoatCheck": "False",
    "GoodForDancing": "False",
    "GoodForKids": "True",
    "GoodForMeal": "{'dessert': False, 'latenight': False, 'lunch': False, 'dinner': True, 'breakfast': False, 'brunch': False}",
    "HappyHour": "False",
    "HasTV": "False",
    "Music": "{'dj': False, 'background_music': True, 'no_music': False, 'karaoke': False, 'live': False, 'video': False, 'jukebox': False}",
    "NoiseLevel": "average",
    "OutdoorSeating": "False",
    "RestaurantsAttire": "casual",
    "RestaurantsDelivery": "True",
    "RestaurantsGoodForGroups": "True",
    "RestaurantsPriceRange2": "2",
    "RestaurantsReservations": "True",
    "RestaurantsTableService": "True",
    "RestaurantsTakeOut": "True",
    "Smoking": "no"
  },
  "categories": "Nightlife, Wine Bars, Indian, Restaurants, Bars",
  "hours": {
    "Monday": "17:0-22:0",
    "Tuesday": "17:0-22:0",
    "Wednesday": "17:0-22:0",
    "Thursday": "17:0-22:0",
    "Friday": "17:30-22:30",
    "Saturday": "17:30-22:30",
    "Sunday": "17:30-22:30"
  }
}
```

### File 5: ```yelp_val_in.csv```
This is the subset of the original dataset that is used as a test file. It is the same as the ```yelp_val.csv``` file except the last column (ratings). This file resembles the format of the test file for the program.<br/>
The file contains records of reviews by users on businesses in the form of stars. A small sample of the dataset is given below.
```
user_id,business_id,stars
wf1GqnKQuvH-V3QN80UOOQ,fThrN4tfupIGetkrz18JOg
39FT2Ui8KUXwmUt6hnwy-g,uW6UHfONAmm8QttPkbMewQ
7weuSPSSqYLUFga6IYP4pg,IhNASEZ3XnBHmuuVnWdIwA
CqaIzLiWaa-lMFYBAsYQxw,G859H6xfAmVLxbzQgipuoA
yy7shAsNWRbGg-8Y67Dzag,rS39YnrhoXmPqHLzCBjeqw
Uk1UKBIAwOqhjZdLm3r9zg,5CJL_2-XwCGBmOav4mFdYg
x-8ZMKKNycT3782Kqf9loA,jgtWfJCJZty_Nctqpdtp3g
0FVcoJko1kfZCrJRfssfIA,JVK8szNDoy9MNiYSz_MiAA
LcCRMIDz1JgshpPGYfLDcA,t19vb_4ML2dg5HZ-MF3muA
C__1BHWTGBNA5s2ZPH289g,h_UvnQfe1cuVICly_kIqHg
```
