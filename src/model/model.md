# Model
In here we are going to introduce our model.
## Setting
* Input: Comes from csv files
* Output:  Popularity or stream_value
    * We formulate our output with the following format:
        * Vector (with n bins. Each bins corresponding with a certain interval) -> (5)
        [46-61,61-67,67-72,72-78,78-100] -> 每個class 大概有1754個
## Data Process
* 我們只保留numerical的資料
* Fill up mean value into Nan entry

## Method
The following are the methods that we have try:
* RandomForest:

* DNN (Deep Neural Network):

## Metric
We use F1-macro to evaluate the performance of our model.

