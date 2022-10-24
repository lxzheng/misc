# Stock Buy Sell Prediction Using Convolutional Neural Network
![](https://github.com/lxzheng/misc/blob/main/images/10-24-2022,%2012-58-28/b210e688-d016-4039-a11e-e5b46824076d.png?raw=true)

Inspired from Research Paper titled ‘Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion’
--------------------------------------------------------------------------------------------------------------------------------------------

This project is loosely based on a research paper titled “[_Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach_](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach)”. I say ‘loosely’ because although I have borrowed the core idea from the paper, there are some things that I have done (or had to do) different as we will see later. The link I have shared above is a preprint of the paper. The paid/main paper may have more details. This paper was suggested by one of the readers of my [previous article](https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944) on stock price prediction and it immediately caught my attention. **Here is the link to the Github** [**repo**](https://github.com/paranoiac-coder/stock_cnn_blog_pub) **and main training notebook on** [**Kaggle**](https://www.kaggle.com/darkknight91/predicting-stock-buy-sell-signal-using-cnn/).

_There is one thing I would like the readers to know — I am not here to claim that I have a ready to use trading model (although I am exploring this method further for my personal use). The idea of converting a conventional tabular or time-series data to image, and training a classification model on it, just seemed too exciting to resist from trying it out and sharing it with the community. Like my_ [_previous article_](https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944) _this is an account of my experience with the project._

1\. What does the research paper say?
-------------------------------------

In this section I will explain the idea presented in the paper. I will discuss the code and implementation in the next section.

> The idea is fairly simple: Calculate 15 [technical indicators](https://www.investopedia.com/terms/t/technicalindicator.asp) with 15 different period lengths (explained below) for each day in your trading data. Then convert the 225 (15\*15) new features into 15x15 images. Label the data as buy/sell/hold based the algorithm provided in the paper. Then train a Convolutional Neural Network like any other image classification problem.

**Feature Engineering:** If you are not aware of what a technical indicator is, I would suggest you check the link provided above. I would explain the concept of technical indicators and time period with a [Simple Moving Average (SMA)](https://www.investopedia.com/terms/s/sma.asp) since it’s simpler. This should be enough for you to understand the idea.

A moving average for a list of numbers is like arithmetic average but instead of calculating the average of all the numbers, we calculate the average of the first ’n’ numbers (n is referred as window size or time period) and then move (or slide) the window by 1 index, thus excluding the first element and including the n+1 element and calculate their average. This process continues. Here is an example to drive this point home:

![](https://github.com/lxzheng/misc/blob/main/images/10-24-2022,%2012-58-28/57b590a7-bbff-4615-acca-e2712e18934d.png?raw=true)

SMA example on excel sheet

This is an example of SMA on window size of 6. The SMA of first 6 elements is shown in orange. Now consider the first column above as the close price of your chosen stock. Now calculate SMA on close price for 14 other window sizes (7 to 20) concatenated on right side of sma\_6. Now you have 15 new features for each row of your dataset. Repeat this process for 14 other technical indicators and drop the null rows.

Some of the indicators used are extensions of SMA. For instance, WMA (Weighted Moving Average) is the average of previous ’n’ days with more weight given to the recent past days. Similarly HMA (Hull Moving Average) is an extension of WMA with following steps:

![](https://github.com/lxzheng/misc/blob/main/images/10-24-2022,%2012-58-28/4011a18d-5ff7-4ddf-9935-3f2ccfbc755a.png?raw=true)

Image source: [Fidelity.com](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/hull-moving-average)

Now you have 225 new features. If you reshape these numbers into a 15x15 array, you have an image! (Albeit, at this point, it’s a single channel. More on this later). **There is one thing to keep in mind though. While constructing these images we should keep the related technical indicators spatially close.** The intuition is, when training for human face recognition, you would not label a picture as human face if it has one eye below the nose. Related pixels should be close by. I am not posting the code to calculate all the indicators for brevity. You can find them in utils.py file.

**Labeling:** What’s left now is to label this dataset. For that, the authors used following algorithm:

![](https://github.com/lxzheng/misc/blob/main/images/10-24-2022,%2012-58-28/3f75ff53-e880-461f-928e-7416578d18fc.png?raw=true)

Algorithm used to label the dataset as buy/sell/hold

At first glance, it may seem formidable, but all it says is this: use a window of 11 days on close price. If the middle number is maximum within the window, label the middle day as ‘sell’ or, if the middle number is minimum then label the middle day as ‘buy’, else label as ‘hold’. Slide the window like explained earlier and repeat. **The idea is to buy at troughs and sell at crests for any 11 day window.** The competency of this algorithm is a different matter and I will get into that toward the end.

**Training:** Authors have used rolling window training, which is similar to the sliding window concept we saw above. If you have stock history data for the year 2000 to 2019 and you decide to train on 5 years data and test on 1 year data then, slice the data for 2000–2004 from dataset for training and 2005 year’s data for testing. Train and test your model on this data. Next select 2001–2005 as training data and 2006 as test data. Use the same model to retrain on this data. Repeat until you reach the end.

**Computational Performance Evaluation:** Authors have provided two types of model evaluations in the paper, computational and financial evaluation. Computational evaluation includes confusion matrix, F1 score, class wise precision etc. Financial evaluation is done by applying the model prediction to real world trading and measure the profit made. I will only discuss the computational evaluation. Financial evaluation can be done by either real world trading or backtesting on held out data, which I may discuss in the future articles.

2\. Implementation
------------------

As mentioned at the beginning of this article, I have not followed the research paper strictly because it didn’t produce expected results. I will mention the differences as and when they come up. But with the changes I made the result was at par with the paper or better in some cases.

The data processing related code can be found in **data\_generator.py**

**Data Source:** I usually get stock data from [Alpha Vantage](https://www.alphavantage.co/) which provides historical stock data for free. I had used it for my previous [project](https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944) as well. Here is how you can download the data.

```
url = "[https://www.alphavantage.co/query?function=TIME\_SERIES\_DAILY\_ADJUSTED&outputsize=full&apikey=api\_key&datatype=csv&symbol=company\_code](https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&apikey=api_key&datatype=csv&symbol=company_code)"  
urllib.request.urlretrieve(url, path\_to\_save)
```

Data looks like this:

![](https://github.com/lxzheng/misc/blob/main/images/10-24-2022,%2012-58-28/39e98ea9-ebe7-4fbe-abe6-4b861e960600.png?raw=true)

**Feature Engineering:** The first deviation from the paper is the technical indicators I used. I couldn’t find library/implementation for some of the indicators that were mentioned in the paper, like PSI. Some indicators were just not clear; for example, [PPO](https://www.investopedia.com/terms/p/ppo.asp) is calculated using [EMA](https://www.investopedia.com/terms/e/ema.asp) of period 12 and 26. How can we calculate PPO for different periods? I tried to use most of the indicators mentioned in the paper for which I found open source implementations to avoid any programming errors. I have implemented some indicators like WMA, HMA, etc, although they are slow and need optimization. Since I have to run it only once and save the data, it’s not an issue for me. You can use different indicators of your choice though. They have also adjusted the prices (open, high, low etc) with adjust ratio. But I haven’t followed this one because I couldn’t find any reference on how to do that adjustment. All the functions for constructing these features are in **utils.py** file.

**Labeling the data:** For this blog, I have used the original labeling algorithm that the authors have used. Here is a direct implementation of it:

The dataset looks like this after feature construction and labeling:

![](https://github.com/lxzheng/misc/blob/main/images/10-24-2022,%2012-58-28/da32deb7-ae9f-47c3-98f0-84a2c7b6dc7f.png?raw=true)

**Normalization:** I used MinMaxScaler from Sklearn to normalize the data in the range of \[0, 1\], although the paper used \[-1, 1\] range (second deviation). This is just a personal preference.

**Feature Selection:** After calculating these indicators, grouping them in the image based on their types (momentum, oscillator, etc), and training many CNN architectures, I realized the model just isn’t learning enough. Maybe the features weren’t good enough. So I decided to go with many other indicators without strictly following the rule of calculating them with different periods. Then I used feature selection technique to chose 225 high-quality features. In fact, I used two feature selection methods f\_classif and mutual\_info\_classif and chose the common features from both of their results. There is no mention of feature selection in the original paper, so third deviation.

At the end I am sorting indices list found intersection of both f\_classif and mutual\_info\_classif. This is to ensure that related features are in close proximity in the image, since I had appended similar type of indicators closely. Feature selection significantly improved the performance of the model.

**Reshaping the data as image:** As of now we have a tabular data with 225 features. We need to convert it as images like this:

This is what the images look like:

![](https://github.com/lxzheng/misc/blob/main/images/10-24-2022,%2012-58-28/d9a90470-d678-4a8f-9da9-3cc95b984f1a.png?raw=true)

Training images

**Handling Class Imbalance:** One more reason why these kinds of problems are tricky to solve is that data is massively imbalanced. Number of instances of ‘hold’ action will always be _much_ greater than buy/sell. In fact the labeling algorithm presented in the paper produces somewhat generous number of buy/sell instances. Any other real world strategy would produce much fewer instances. And to further complicate things, classification of ‘hold’ event would not be straight forward (more on this towards the end).

![](https://github.com/lxzheng/misc/blob/main/images/10-24-2022,%2012-58-28/cc13a0a4-d58b-4fdc-8729-bd5c0d57f209.png?raw=true)

This is really less for model to learn anything significant. The paper mentions only “resampling” as a way of tackling this problem. I tried oversampling, synthetic data generation (SMOTE, ADASYN) but none of them gave any satisfactory result. Finally I settled for “sample weights”, wherein you tell the model to pay more attention to some samples (fourth deviation). This comes handy while dealing with class imbalance. Here is how you can calculate sample weight:

This array of sample weights is then passed to Keras ‘fit’ function. You can also look into ‘class\_weights’ parameter.

**Training:** All the training related code can be found in “**stock\_keras.ipynb**”. The model architecture mentioned in the paper had some points missing. For example, they didn’t mention the strides they had used. But trying with stride=1 and padding=same, I realized the model was just too big, especially for training on 5 years of data. I didn’t have any luck with sliding window training no matter how small a network I used. So I trained with full training data with cross validation (fifth deviation). But I have included the code for sliding/rolling window training in the project (in “train.py” file). So, I used a very similar model with small differences like dropouts etc. This is the model I trained with (I have not tried extensive hyperparameter tuning):

![](https://github.com/lxzheng/misc/blob/main/images/10-24-2022,%2012-58-28/7f21e60a-9831-408f-b41b-ef29fc798df4.png?raw=true)

Keras model training was done with EarlyStopping and ReduceLROnPlateau callbacks like this:

![](https://github.com/lxzheng/misc/blob/main/images/10-24-2022,%2012-58-28/147d9192-1f1c-468c-be92-bc00585a9852.png?raw=true)

As you can see above I have used F1 score as metric. For test data evaluation I have also used confusion matrix, Sklearn’s weighted F1 score and Kappa (which I got to know about recently, have to dig deeper).

On Walmart data the above model gave the following result:

![](https://github.com/lxzheng/misc/blob/main/images/10-24-2022,%2012-58-28/feafce75-e2dc-451a-90cb-a4c200954f5e.png?raw=true)

This result somewhat varies every time I run it, which may be due to Keras weight initialization. This is actually a known behavior, with a long thread of discussions [here](https://github.com/keras-team/keras/issues/2743). In short you have to set random seed for both numpy and tensorflow. I have set random seed for numpy only. So I am not sure if it will fix this issue. I will update here once I try it out. But most of the time and for most other CNN architectures I have tried, precision of class 0 and class 1 (buy/sell) is less than class 2 (with class 0/1 being 70s).

The authors got following results:

![](https://github.com/lxzheng/misc/blob/main/images/10-24-2022,%2012-58-28/dbcd4c48-a5a0-4c37-b70e-b282ed4e6fe0.png?raw=true)

Result for Dow-30 presented in the paper

If you notice, “hold” class scores are significantly worse that “buy/sell”, both in our result and the paper’s. I think this result is quite promising given that model can identify most of the buy/sell instances. Here is what the authors have to say about it:

> “However, a lot of false entry and exit points are also generated. This is mainly due to the fact that “Buy” and “Sell” points appear much less frequent than “Hold” points, it is not easy for the neural network to catch the “seldom” entry and exit points without jeopardizing the general distribution of the dominant “Hold” values. In other words, in order to be able to catch most of the “Buy” and “Sell” points (recall), the model has a trade-off by generating false alarms for non-existent entry and exit points (precision). Besides, Hold points are not as clear as “Buy” and “Sell” (hills and valleys). It is quite possible for the neural network to confuse some of the “Hold” points with “Buy” and “Sell” points, especially if they are close to the top of the hill or bottom of the valley on sliding windows.”

3\. Further Improvements
------------------------

*   There is definitely a lot of room for better network architecture and hyperparameter tuning.
*   Using CNN with same architecture on other datasets didn’t give as impressive precision for buy and sell. But by playing around with hyperparameters we can definitely improve it to similar figures as Walmart.
*   Although these results seem good enough, there is no guarantee that it would give you profits on real world trading because it would be limited by the strategy you choose to label your data. For example, I backtested above trading strategy (with original labels and not model predictions!) but I didn’t make much profit. But that depends on the labeling of the data. If someone uses a better strategy to label the training data, it may perform better.
*   Exploring other technical indicators may further improve the result.

4\. Conclusion
--------------

I started working on this project with a very skeptical mind. I was not sure if the images would have enough information/patterns for the ConvNet to find. But since the results seem to be much better than random prediction, this approach seems promising. I especially loved the way they converted the time series problem to image classification.

**UPDATE- 12/7/2020:** Major update- There was a bug in label creation, which was assigning labels to last day of the window instead of middle item. I have also a updated this article with new results. New model updated in “stock\_keras.ipynb”

Code fix is available on GitHub as well. Please note that since I have moved to PyTorch and I don’t have a working Tensorflow environment anymore, I trained this model on cloud and had to copy paste the fixes. So, I couldn’t test the final code completely (the training part).

Inserted the code gists which were missing due changes to my GitHub account.

**UPDATE- 23/2/2020:** I have just discovered a bug in my model creation function “create\_model\_cnn”, where I use the following check to add MaxPool layers:

```
if params\["conv2d\_layers"\]\['conv2d\_mp\_1'\] == 1  
replace this with  
if params\["conv2d\_layers"\]\['conv2d\_mp\_1'\] >= 0
```

Do the same for “conv2d\_mp\_2” as well. There is nothing wrong with the model or program as such, it’s just that I had been exploring the hyperparameters search space without any MaxPools :-( . Need to explore if model can perform better with MaxPool layers.

**UPDATE- 09/02/2020**: Added explanation for some of the more complicated technical indicators in “Feature Engineering” section.