
# Introduction
Stock price is generally considered as a random walk process according to Paul Samuelson's theory in 1965 [1]. Despite this fact, machine learning is still giving people a guide.
In this work, I study four machine learning methods (SVM, Rand Forest, kNN, Recurrent NN) to find the best approach to make a good prediction. The input is purely based on historical OHLC and volume of nine ETFs ('XLE', 'XLU', 'XLK', 'XLB', 'XLP', 'XLY', 'XLI', 'XLV', 'SPY'). Though the performance isn't existing in terms of fortune making, the result shows some positive direction toward future study.

# Approach
The problem of predicting next day's price is transformed into: predict `Up` if tomorrow's close is greater than open; predict `Down` if tomorrow's close is smaller than open. Thus, I transform this to be a classification problem: using today's OHLC to predict next day's `Up/Down`.

The buying strategy will follow the prediction result, if `Up`, then `long`, otherwise `short`.

## Feature generation
Raw data contains five columns: Open, High, Low, Close, and Volume. To enrich information,
1. add one extra column: `Return` (= (Close - Open) / Open)
2. by user defined `delta`, append `delta` days' OHLC, volume, returns
3. also append `delta` days' `close price percentage change` and `rolling mean of returns`

## Parameter tuning
We need to find the best of two parameter sets: `delta` and its corresponding parameters for classifier.

This is tuned by a series of cross validation process. Following is the best `delta` found in this study.

|SVM | Rand Forest | kNN | Recurrent NN |
| --- | --- | --- | -- |
| 4 | 3 | 99 | 20 |