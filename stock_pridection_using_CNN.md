This page is scraped from web using github writer plugin. 

Source: https://towardsdatascience.com/stock-market-action-prediction-with-convnet-8689238feae3

# **Stock Buy/Sell Prediction Using Convolutional Neural Network**

## Inspired from Research Paper titled ‘Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion’

This project is loosely based on a research paper titled “[_Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach_](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach)”. I say ‘loosely’ because although I have borrowed the core idea from the paper, there are some things that I have done (or had to do) different as we will see later. The link I have shared above is a preprint of the paper. The paid/main paper may have more details. This paper was suggested by one of the readers of my [previous article](https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944) on stock price prediction and it immediately caught my attention. **Here is the link to the Github** [**repo**](https://github.com/paranoiac-coder/stock_cnn_blog_pub) **and main training notebook on** [**Kaggle**](https://www.kaggle.com/darkknight91/predicting-stock-buy-sell-signal-using-cnn/).

_There is one thing I would like the readers to know — I am not here to claim that I have a ready to use trading model (although I am exploring this method further for my personal use). The idea of converting a conventional tabular or time-series data to image, and training a classification model on it, just seemed too exciting to resist from trying it out and sharing it with the community. Like my_ [_previous article_](https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944) _this is an account of my experience with the project._

## **1\. What does the research paper say?**

In this section I will explain the idea presented in the paper. I will discuss the code and implementation in the next section.

> _The idea is fairly simple: Calculate 15_ [_technical indicators_](https://www.investopedia.com/terms/t/technicalindicator.asp) _with 15 different period lengths (explained below) for each day in your trading data. Then convert the 225 (15\*15) new features into 15x15 images. Label the data as buy/sell/hold based the algorithm provided in the paper. Then train a Convolutional Neural Network like any other image classification problem._

**Feature Engineering:** If you are not aware of what a technical indicator is, I would suggest you check the link provided above. I would explain the concept of technical indicators and time period with a [Simple Moving Average (SMA)](https://www.investopedia.com/terms/s/sma.asp) since it’s simpler. This should be enough for you to understand the idea.

A moving average for a list of numbers is like arithmetic average but instead of calculating the average of all the numbers, we calculate the average of the first ’n’ numbers (n is referred as window size or time period) and then move (or slide) the window by 1 index, thus excluding the first element and including the n+1 element and calculate their average. This process continues. Here is an example to drive this point home:

![](https://miro.medium.com/max/428/1*xoW3xZFNCgpaPcZjcKm0sw.png)

SMA example on excel sheet

This is an example of SMA on window size of 6. The SMA of first 6 elements is shown in orange. Now consider the first column above as the close price of your chosen stock. Now calculate SMA on close price for 14 other window sizes (7 to 20) concatenated on right side of sma\_6. Now you have 15 new features for each row of your dataset. Repeat this process for 14 other technical indicators and drop the null rows.

Some of the indicators used are extensions of SMA. For instance, WMA (Weighted Moving Average) is the average of previous ’n’ days with more weight given to the recent past days. Similarly HMA (Hull Moving Average) is an extension of WMA with following steps:

![](https://miro.medium.com/max/1362/1*MVxfxTQcTI9MYdGtpAs4Gg.png)

Image source: [Fidelity.com](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/hull-moving-average)

Now you have 225 new features. If you reshape these numbers into a 15x15 array, you have an image! (Albeit, at this point, it’s a single channel. More on this later). **There is one thing to keep in mind though. While constructing these images we should keep the related technical indicators spatially close.** The intuition is, when training for human face recognition, you would not label a picture as human face if it has one eye below the nose. Related pixels should be close by. I am not posting the code to calculate all the indicators for brevity. You can find them in utils.py file.

**Labeling:** What’s left now is to label this dataset. For that, the authors used following algorithm:

![Algorithm used to label the dataset as buy/sell/hold](https://miro.medium.com/max/1362/1*WghX_EwKnTEuPQIJE0I36Q.png)

Algorithm used to label the dataset as buy/sell/hold

At first glance, it may seem formidable, but all it says is this: use a window of 11 days on close price. If the middle number is maximum within the window, label the middle day as ‘sell’ or, if the middle number is minimum then label the middle day as ‘buy’, else label as ‘hold’. Slide the window like explained earlier and repeat. **The idea is to buy at troughs and sell at crests for any 11 day window.** The competency of this algorithm is a different matter and I will get into that toward the end.

**Training:** Authors have used rolling window training, which is similar to the sliding window concept we saw above. If you have stock history data for the year 2000 to 2019 and you decide to train on 5 years data and test on 1 year data then, slice the data for 2000–2004 from dataset for training and 2005 year’s data for testing. Train and test your model on this data. Next select 2001–2005 as training data and 2006 as test data. Use the same model to retrain on this data. Repeat until you reach the end.

**Computational Performance Evaluation:** Authors have provided two types of model evaluations in the paper, computational and financial evaluation. Computational evaluation includes confusion matrix, F1 score, class wise precision etc. Financial evaluation is done by applying the model prediction to real world trading and measure the profit made. I will only discuss the computational evaluation. Financial evaluation can be done by either real world trading or backtesting on held out data, which I may discuss in the future articles.

## **2\. Implementation**

As mentioned at the beginning of this article, I have not followed the research paper strictly because it didn’t produce expected results. I will mention the differences as and when they come up. But with the changes I made the result was at par with the paper or better in some cases.

The data processing related code can be found in **data\_generator.py**

**Data Source:** I usually get stock data from [Alpha Vantage](https://www.alphavantage.co/) which provides historical stock data for free. I had used it for my previous [project](https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944) as well. Here is how you can download the data.

url = "[https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&apikey=api_key&datatype=csv&symbol=company_code](https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&apikey=api_key&datatype=csv&symbol=company_code)"  
urllib.request.urlretrieve(url, path\_to\_save)

Data looks like this:

![](https://miro.medium.com/max/1362/1*mhywNu-hVPjXNFqrsOwTFQ.png)

**Feature Engineering:** The first deviation from the paper is the technical indicators I used. I couldn’t find library/implementation for some of the indicators that were mentioned in the paper, like PSI. Some indicators were just not clear; for example, [PPO](https://www.investopedia.com/terms/p/ppo.asp) is calculated using [EMA](https://www.investopedia.com/terms/e/ema.asp) of period 12 and 26. How can we calculate PPO for different periods? I tried to use most of the indicators mentioned in the paper for which I found open source implementations to avoid any programming errors. I have implemented some indicators like WMA, HMA, etc, although they are slow and need optimization. Since I have to run it only once and save the data, it’s not an issue for me. You can use different indicators of your choice though. They have also adjusted the prices (open, high, low etc) with adjust ratio. But I haven’t followed this one because I couldn’t find any reference on how to do that adjustment. All the functions for constructing these features are in **utils.py** file.

**Labeling the data:** For this blog, I have used the original labeling algorithm that the authors have used. Here is a direct implementation of it:

```python
def create_labels(self, df, col_name, window_size=11):
        """
        Data is labeled as per the logic in research paper
        Label code : BUY => 1, SELL => 0, HOLD => 2
        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy
        returns : numpy array with integer codes for labels with
                  size = total-(window_size)+1
        """

        self.log("creating label with original paper strategy")
        row_counter = 0
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        print("Calculating labels")
        pbar = tqdm(total=total_rows)

        while row_counter < total_rows:
            if row_counter >= window_size - 1:
                window_begin = row_counter - (window_size - 1)
                window_end = row_counter
                window_middle = (window_begin + window_end) / 2

                min_ = np.inf
                min_index = -1
                max_ = -np.inf
                max_index = -1
                for i in range(window_begin, window_end + 1):
                    price = df.iloc[i][col_name]
                    if price < min_:
                        min_ = price
                        min_index = i
                    if price > max_:
                        max_ = price
                        max_index = i

                if max_index == window_middle:
                    labels[window_middle] = 0
                elif min_index == window_middle:
                    labels[window_middle] = 1
                else:
                    labels[window_middle] = 2

            row_counter = row_counter + 1
            pbar.update(1)

        pbar.close()
        return labels
```

The dataset looks like this after feature construction and labeling:

![](https://miro.medium.com/max/1362/1*gsCux9rcKNZoy0KPkXKKgw.png)

**Normalization:** I used MinMaxScaler from Sklearn to normalize the data in the range of \[0, 1\], although the paper used \[-1, 1\] range (second deviation). This is just a personal preference.

```python
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from collections import Counter

list_features = list(df.loc[:, 'open':'eom_26'].columns)
print('Total number of features', len(list_features))
x_train, x_test, y_train, y_test = train_test_split(df.loc[:, 'open':'eom_26'].values, df['labels'].values, train_size=0.8, 
                                                    test_size=0.2, random_state=2, shuffle=True, stratify=df['labels'].values)

if 0.7*x_train.shape[0] < 2500:
    train_split = 0.8
else:
    train_split = 0.7

print('train_split =',train_split)
x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, train_size=train_split, test_size=1-train_split, 
                                                random_state=2, shuffle=True, stratify=y_train)
mm_scaler = MinMaxScaler(feature_range=(0, 1)) # or StandardScaler?
x_train = mm_scaler.fit_transform(x_train)
x_cv = mm_scaler.transform(x_cv)
x_test = mm_scaler.transform(x_test)

print("Shape of x, y train/cv/test {} {} {} {} {} {}".format(x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_test.shape, y_test.shape))
```

**Feature Selection:** After calculating these indicators, grouping them in the image based on their types (momentum, oscillator, etc), and training many CNN architectures, I realized the model just isn’t learning enough. Maybe the features weren’t good enough. So I decided to go with many other indicators without strictly following the rule of calculating them with different periods. Then I used feature selection technique to chose 225 high-quality features. In fact, I used two feature selection methods f\_classif and mutual\_info\_classif and chose the common features from both of their results. There is no mention of feature selection in the original paper, so third deviation.

```python
from operator import itemgetter
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

num_features = 225  # should be a perfect square
selection_method = 'all'
topk = 320 if selection_method == 'all' else num_features

if selection_method == 'anova' or selection_method == 'all':
    select_k_best = SelectKBest(f_classif, k=topk)
    if selection_method != 'all':
        x_train = select_k_best.fit_transform(x_main, y_train)
        x_cv = select_k_best.transform(x_cv)
        x_test = select_k_best.transform(x_test)
    else:
        select_k_best.fit(x_main, y_train)
    
    selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)
    print(selected_features_anova)
    print(select_k_best.get_support(indices=True))
    print("****************************************")
    
if selection_method == 'mutual_info' or selection_method == 'all':
    select_k_best = SelectKBest(mutual_info_classif, k=topk)
    if selection_method != 'all':
        x_train = select_k_best.fit_transform(x_main, y_train)
        x_cv = select_k_best.transform(x_cv)
        x_test = select_k_best.transform(x_test)
    else:
        select_k_best.fit(x_main, y_train)

    selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)
    print(len(selected_features_mic), selected_features_mic)
    print(select_k_best.get_support(indices=True))
    
if selection_method == 'all':
    common = list(set(selected_features_anova).intersection(selected_features_mic))
    print("common selected featues", len(common), common)
    if len(common) < num_features:
        raise Exception('number of common features found {} < {} required features. Increase "topk variable"'.format(len(common), num_features))
    feat_idx = []
    for c in common:
        feat_idx.append(list_features.index(c))
    feat_idx = sorted(feat_idx[0:225])
    print(feat_idx)  # x_train[:, feat_idx] will give you training data with desired features
```

At the end I am sorting indices list found intersection of both f\_classif and mutual\_info\_classif. This is to ensure that related features are in close proximity in the image, since I had appended similar type of indicators closely. Feature selection significantly improved the performance of the model.

**Reshaping the data as image:** As of now we have a tabular data with 225 features. We need to convert it as images like this:

```
dim = int(np.sqrt(num_features))
x_train = reshape_as_image(x_train, dim, dim)
x_cv = reshape_as_image(x_cv, dim, dim)
x_test = reshape_as_image(x_test, dim, dim)
# adding a 1-dim for channels (3)
x_train = np.stack((x_train,) * 3, axis=-1)
x_test = np.stack((x_test,) * 3, axis=-1)
x_cv = np.stack((x_cv,) * 3, axis=-1)
print("final shape of x, y train/test {} {} {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
```

This is what the images look like:

![](https://miro.medium.com/max/1362/1*frO4YPM-WiAMGmLIqDSfSw.png)

Training images

**Handling Class Imbalance:** One more reason why these kinds of problems are tricky to solve is that data is massively imbalanced. Number of instances of ‘hold’ action will always be _much_ greater than buy/sell. In fact the labeling algorithm presented in the paper produces somewhat generous number of buy/sell instances. Any other real world strategy would produce much fewer instances. And to further complicate things, classification of ‘hold’ event would not be straight forward (more on this towards the end).

![](https://miro.medium.com/max/1362/1*LLb8lPcn1pWrEOA9TLu94w.png)

This is really less for model to learn anything significant. The paper mentions only “resampling” as a way of tackling this problem. I tried oversampling, synthetic data generation (SMOTE, ADASYN) but none of them gave any satisfactory result. Finally I settled for “sample weights”, wherein you tell the model to pay more attention to some samples (fourth deviation). This comes handy while dealing with class imbalance. Here is how you can calculate sample weight:

```python
def get_sample_weights(self, y):
        """
        calculate the sample weights based on class weights. Used for models with
        imbalanced data and one hot encoding prediction.
        params:
            y: class labels as integers
        """

        y = y.astype(int)  # compute_class_weight needs int labels
        class_weights = compute_class_weight('balanced', np.unique(y), y)

        print("real class weights are {}".format(class_weights), np.unique(y))
        print("value_counts", np.unique(y, return_counts=True))
        sample_weights = y.copy().astype(float)
        for i in np.unique(y):
            sample_weights[sample_weights == i] = class_weights[i]  # if i == 2 else 0.8 * class_weights[i]
            # sample_weights = np.where(sample_weights == i, class_weights[int(i)], y_)

        return sample_weights
```

This array of sample weights is then passed to Keras ‘fit’ function. You can also look into ‘class\_weights’ parameter.

**Training:** All the training related code can be found in “**stock\_keras.ipynb**”. The model architecture mentioned in the paper had some points missing. For example, they didn’t mention the strides they had used. But trying with stride=1 and padding=same, I realized the model was just too big, especially for training on 5 years of data. I didn’t have any luck with sliding window training no matter how small a network I used. So I trained with full training data with cross validation (fifth deviation). But I have included the code for sliding/rolling window training in the project (in “train.py” file). So, I used a very similar model with small differences like dropouts etc. This is the model I trained with (I have not tried extensive hyperparameter tuning):

![](https://miro.medium.com/max/706/1*ozmL77jKHHeYyIwbzatCOQ.png)

Keras model training was done with EarlyStopping and ReduceLROnPlateau callbacks like this:

![](https://miro.medium.com/max/1362/1*jLWJz2VEKhgDcBJdrv7LOA.png)

As you can see above I have used F1 score as metric. For test data evaluation I have also used confusion matrix, Sklearn’s weighted F1 score and Kappa (which I got to know about recently, have to dig deeper).

On Walmart data the above model gave the following result:

![](https://miro.medium.com/max/784/1*YLKDhlcx6TtzIoVSdJ2xNw.png)

This result somewhat varies every time I run it, which may be due to Keras weight initialization. This is actually a known behavior, with a long thread of discussions [here](https://github.com/keras-team/keras/issues/2743). In short you have to set random seed for both numpy and tensorflow. I have set random seed for numpy only. So I am not sure if it will fix this issue. I will update here once I try it out. But most of the time and for most other CNN architectures I have tried, precision of class 0 and class 1 (buy/sell) is less than class 2 (with class 0/1 being 70s).

The authors got following results:

![](https://miro.medium.com/max/1362/1*Vzn12FzkVfOcpMXkCoCWAw.png)

Result for Dow-30 presented in the paper

If you notice, “hold” class scores are significantly worse that “buy/sell”, both in our result and the paper’s. I think this result is quite promising given that model can identify most of the buy/sell instances. Here is what the authors have to say about it:

> “However, a lot of false entry and exit points are also generated. This is mainly due to the fact that “Buy” and “Sell” points appear much less frequent than “Hold” points, it is not easy for the neural network to catch the “seldom” entry and exit points without jeopardizing the general distribution of the dominant “Hold” values. In other words, in order to be able to catch most of the “Buy” and “Sell” points (recall), the model has a trade-off by generating false alarms for non-existent entry and exit points (precision). Besides, Hold points are not as clear as “Buy” and “Sell” (hills and valleys). It is quite possible for the neural network to confuse some of the “Hold” points with “Buy” and “Sell” points, especially if they are close to the top of the hill or bottom of the valley on sliding windows.”

## **3\. Further Improvements**

*   There is definitely a lot of room for better network architecture and hyperparameter tuning.
*   Using CNN with same architecture on other datasets didn’t give as impressive precision for buy and sell. But by playing around with hyperparameters we can definitely improve it to similar figures as Walmart.
*   Although these results seem good enough, there is no guarantee that it would give you profits on real world trading because it would be limited by the strategy you choose to label your data. For example, I backtested above trading strategy (with original labels and not model predictions!) but I didn’t make much profit. But that depends on the labeling of the data. If someone uses a better strategy to label the training data, it may perform better.
*   Exploring other technical indicators may further improve the result.

## **4\. Conclusion**

I started working on this project with a very skeptical mind. I was not sure if the images would have enough information/patterns for the ConvNet to find. But since the results seem to be much better than random prediction, this approach seems promising. I especially loved the way they converted the time series problem to image classification.

**UPDATE- 12/7/2020:** Major update- There was a bug in label creation, which was assigning labels to last day of the window instead of middle item. I have also a updated this article with new results. New model updated in “stock\_keras.ipynb”

Code fix is available on GitHub as well. Please note that since I have moved to PyTorch and I don’t have a working Tensorflow environment anymore, I trained this model on cloud and had to copy paste the fixes. So, I couldn’t test the final code completely (the training part).

Inserted the code gists which were missing due changes to my GitHub account.

**UPDATE- 23/2/2020:** I have just discovered a bug in my model creation function “create\_model\_cnn”, where I use the following check to add MaxPool layers:

if params\["conv2d\_layers"\]\['conv2d\_mp\_1'\] == 1  
replace this with  
if params\["conv2d\_layers"\]\['conv2d\_mp\_1'\] >= 0

Do the same for “conv2d\_mp\_2” as well. There is nothing wrong with the model or program as such, it’s just that I had been exploring the hyperparameters search space without any MaxPools :-( . Need to explore if model can perform better with MaxPool layers.

**UPDATE- 09/02/2020**: Added explanation for some of the more complicated technical indicators in “Feature Engineering” section.
