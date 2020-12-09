# Machine Learning and Quantitative Trading - Session 5
**Where are we now**
 - Get the data (done)
 - Store the data (done)
 - Transform data into training set
 - Building predictive models
 - Building event driving back-testing pipelines

**Current Session Content**
- Data processing (raw data clean)
- Prepare training set from cleaned data
- Building predictive model

**Warning:** Building Models vs Building (event driving) trading strategiesI
In this session, we focus on building models at first.
# How to set up training dataset
|t| o | c | h | l |
|1|--|
|2 |  |
|3 |  |
|$t_0$ |  |

We get some data attributes from online, how can we built the training data which is understandable by the machine learning? You need transform the data which before $t_0$ to be the features, and y is the predicted result from the model. 

|t|$x_1$| $x_2$|$x_3$|$x_n$|  Y
|--|--|
| $t_0$ |  |

So the first thing we do is transform the general financial data to supervised machine learning framework. 

In quant analysis/research, most difference are in feature selection and base on features to select suitable model. 

How to choose Features (predictors $x_1...x_n$)
- Choosing the feature is kind of understanding of the questions from the quant
   - Different people with different believes
   - Different believes with different feature selection
   - Selection procedure is the same

## Basic method and princple
- Index/feature selection depends on your understang
eg. the SP&500, which kind of features are relevant? 

### General features
- Time lags
`def create_lagged_series (symbol, start_date, end_date, lags=5)`
This creates a pandas DataFrame that stores the perventage returns of the adjusted closing value of a stock obtained from Yahoo Finance, along with a number of lagged returns from the prior trading days (lags defaults to 5 days). Trading volume, as well as the Direction from the previous day, are also included.
- Normalization of Price and Volumn of open, high, low
  $(P_{open}^{(t)} - P_{open}^{(t-1)}) / P_{open}^{(t-1)}$
  $(P_{high}^{(t-1)} - P_{high}^{(t-2)}) / P_{high}^{(t-2)}$
  $(P_{low}^{(t-1)} - P_{low}^{(t-2)}) / P_{low}^{(t-2)}$
  $(V_{open}^{(t)} - V_{open}^{(t-1)}) / V_{open}^{(t-1)}$
  $(V_{high}^{(t-1)} - V_{high}^{(t-2)}) / V_{high}^{(t-2)}$
  $(V_{low}^{(t-1)} - V_{low}^{(t-2)}) / V_{low}^{(t-2)}$
- Using cross validation to find the $\lambda$, the optimal window size
  $PH_{\lambda}^{(t)} = max_{t-\lambda\leq i\leq t-1} P_{high}^{(i)}$
  $PL_{\lambda}^{(t)} = min_{t-\lambda\leq i\leq t-1} P_{low}^{(i)}$
  $VH_{\lambda}^{(t)} = max_{t-\lambda\leq i\leq t-1} P_{high}^{(i)}$
  $VL_{\lambda}^{(t)} = min_{t-\lambda\leq i\leq t-1} P_{low}^{(i)}$
  Then we can use above features in to following formula
  $X_1 =\frac{P_{open}^{(t)} - P_{optn}^{(t-1)}}{PH_{\lambda}^{(t)} - PL_{\lambda}^{(t)}}$
  
  $X_2 = \frac{V_{open}^{(t)} - V_{optn}^{(t-1)}}{VH_{\lambda}^{(t)} - VL_{\lambda}^{(t)}}$
- Stock Index features
|Feature| Category |
|S&P 500|Stock Index|
| DJLA | Stock Index |
|Nikkei | Stock Index |
| FTSH| Stock Index |
| SSE | Stock Index |
| Crude Oil | Commedity |
|CNYUSD | Currency Rate |
| JPYUSD | Currency Rate |
| EUROUSD | Currency Rate|
|AUDUSD | Currency Rate |
| STI | Stock Index |
|NASDAQ | Stock Index |
| Gold price | Commedity |
- CCI feature
  CCI = (Typical price - MA of Typical price) / (0.015 * Standard deviation of Typical price)
  **MA = Moving average**
    - CCI can be used to determine overbought and oversold levels. Readings above +100 can imply an overbought condition, while readings below -100 can imply an oversold condition. However, one should be a careful because se such overboughta security can continue moving higher after the CCI indicator becomes overbought. Likewise, securities can continue moving lower after the indicator becomes oversold. 
    -  Whenever the security is in overbought/oversold levels as indicated by the CCI, there is a good chance that the price will see corrections. Hence a trader can use such overbought/oversold levels to enter in short/long positions.
    - Traders can also look for divergence signals to take suitable positions using CCI. A bullish divergence occurs when the underlying security makes a lower low and the CCI forms a higher low, which shows less downside momentum. Similarly, a bearish divergence is formed when the security records a higher high and the CCI forms a lower high, which shows less upside momentum.
- Ease of Movement (EVM)
   - Distance moved = ((Current High + Current Low)/2 - (Prior High + Prior Low)/2)
   - Ease of Movement (EMV) is a volume-based oscillator which was developed by Richard Arms. EVM indicates the ease with which the prices rise or fall taking into account the volume of the security. For example, a price rise on a low volume means prices advanced with relative ease, and there was little selling pressure. Positive EVM values imply that the market is moving higher with ease, while negative values indicate an easy decline.
- MA (moving average with $\lambda$)
- Rate of Change (ROC)
   - The Rate of Change (ROC) is a technical indicator that measures the percentage change between the most recent price and price "n" day's ago. The indicator fluctuates around the zero line.
   - If the ROC is rising, it gives a bullish signal, while a falling ROC gives a bearish signal. One can compute ROC based on different periods in order to gauge the short-term momentum or the long-term momentum.
- BB (Bollinger Bands)
   - These bands comprise of an upper bollinger band and a lower bollinger band, and are placed two standard deviations above and below a moving average.
   - Bollinger bands expand and contract based on the volatility. During a period of rising volatility, the bands widen, and they contract as the volatility decreases. Prices are considered to be relatively high when they move above the upper band and relatively low when they go below the lower band.
- Force Index (FI)
  - The force index was created by Alexander Elder. The force index takes into account the direction of the stock price, the extent of the stock price movement, and the volume. Using these three elements it forms an oscillator that measures the buying and the selling pressure.
  - Eash of these three factors plays an important role in determination of the force index. For example, a big advance in prices, which is given by the xtent of the price movement, shows a strong buying pressure. A big decline on heavy volume indicates a strong selling pressure.
- Other more
  - See in [stock technical indicators](https://school.stockcharts.com/doku.php?id=technical_indicators)

## Create training dataset
    # Create a lagged series of the S&P500 US stock market index
    snpret = create_lagged_series(
    	"^GSPC", datetime.datetime(2001,1,10), 
    	datetime.datetime(2005,12,31), lags=5
    )

    # Use the prior two days of returns as predictor 
    # values, with direction as the response
    X = snpret[["Lag1","Lag2"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2005,1,1)

    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

 Normally $x_{train}$ is a matrix with size of $n * d$, and y is a vector with $n*1$

In Skelearn, will the methods could be handled in parallel processing, it has parameter, n_jobs. If you have 8 kernels, then set up n_jobs = -1, it will full processing with 8 kernels. Using DL package, it will automatically help use parallel processing.

Using NLP tech to select the stock is okay, but using it to do trading, the NLP could not treated as single indicator. 

## Modeling
### Random Forests
In most case, it is as useful as deep neural network, the difference between financial data and image data, the feature engineer for financial data is more better, so if using the NN for the data which with good feature, it will be overfitting. Random Forests with good normalized performance, there is math proof that Random Forests with a lot of trees, it will not reduced the error correction. But you increase the neuros and layers in NN, it will be easier overfitting. 

Another difference between DT with NN, the DT is white box model, you will know the decision rule,  if could be equal to the if and else conditions. But single tree with not good result, so we could use ensembling learning, combine the individual tree. The method proof the ensembing will be very good. The NN is black box model.

Here is the tricky for single tree creating:
1) Training dataset selection for each tree in difference
2) Feature selection for each tree in difference, typically we choose  $m \approx \surd p$, p is the total features. 

### Boosting algorithm for regression trees
Xgbost (Gradient based decision boost) is its algo library name, the algo implemented with following steps:
   1. Set $\hat f(x) = 0$ and $r_i = y_i$ for all i in the training set.
   2. For $b = 1,2,..., B$, repeat:
       2.1 Fit a gree $\hat f^b$ with d splits (d+1 teerminal nodes) to the training data (X, r).
       2.2 Update $\hat f$ by adding in a shrunken version of the new tree:
       $\hat f(x) \leftarrow \hat f(x) + \lambda \hat f^b(x).$
        2.3 Update the residuals,
        $r_i \leftarrow r_i - \lambda \hat f^b(x_i).$
    3. Output the boosted model,
        $\hat f(x) = \sum_{b=1}^B \lambda \hat f^b (x).$

Another another is GBDTs, the Random forest is using parallel processing, the GBDTs using the first tree fits the result of $\hat y$, then the 2nd tree fits the residual of ($y - \hat y$), the 3rd tree fits the residual of residul of the ($y - \hat y$). So the gradient decision tree like fourier series decomposition. Currently the best GBDT is xgboost library in python.

Compare advantage and disadvantage with random forest:
1. RF allow parallel processing
2. GBDT with good performance more

Both of them are not easy to be overfitting. Blending, stacking or voting are common marco methods for different kinds of base classifier, eg SVM, Linear Classifer, ensembling together for learning. Difference between Adaboost and Xgboost like java and java script. The marco methods you could image it like dynamic programming, it teachs you how to think about solving the problems, it is not a specific algorithm.  

#### Tuning parameters for boosting
1. The **number of trees** B. Unlike bagging and random forests, boosting can overfit if B is too large, although this overfitting tends to occur slowly if at all. We use corss-validation to select B.
2. The **shrinkage parameter** $\lambda$, a small positive number. This controls the rate at which boosting learns. Typical values are 0.01 or 0.001, annd the right choice can depned on the problem. Very small $\lambda$ can require using a very large value of B in order to achieve good preformance.
3. The **number of splits** d in earch tree, which controls the complexity of the boosted ensemble. Often d=1 works well, in which case each tree is a **stump**, consisting of a single slit and resulting in an additive model. More generally d is the **interaction depth**, and controls the interaction order of the boosted odel, since d splits can involve at most d variables.

**Gaussian regression** will help you analysis the correlation of 2 by 2 features when validate the training model. According to the result of Gaussian regression, the features which are not confident by yourself, you can do further model training to find the best optimal solution. But if your model can be cross validated, then it is not necessary using Gaussian regression.

### Logistic Regression

$F(x) = \frac{1}{1+e^{-(\beta_0+\beta_1x)}}$

It is a genalized linear model, it is used for 0,1 classification. It is used for old NN as active function, now more NN using softmax, tanth, ReLu as active function. Normally logistic regression will be used as model baseline/benchmark.

### LASSO
It is a regression model.

### SVM
It is a classification model.

### SVR
It is a regression model.

### ANN
Using MLP is okay. Keras is better packaged than tensorflow for MLP.  Keras is based on TF and Theano.

BP (backpropogation) is a method to help NN find the value of $w_n$, the key of this method which is using dynamic programming strategy + chain rule of partial derivative, it help reduce multiple times derivative on each parameters, just one time derivative on y and x.

When using Keras to fit the ANN model, we need provide:
-  Loss function
   -  Cross entropy for classification
   -  MSE for regression

Code for ANN model creating - Keras

    model = Sequential()
    # Input with 784 dimensions, first layer with 512 dimensions
    model.add(Dense(512, input_shape-(784,)))
    # Relu as active function for first layer
    model.add(Activation("relu"))
    # 20% parameters will be closed to train the individual NN model, then ensemble those single NN, this is dropout method
    model.add(Dropout(0.2))
    # 2nd layer with 512 dimensions
    model.add(Dense(512))
    model.add(Activation("relu")
    model.add(Dropout(0.2))
    # 3rd layer with 1 dimension
    model.add(Dense(1))
    model.add(Activation("tanh")
    # print out details of each layer
    model.summary()
    # MSE for regression, optimizer has other options: sgd, adam, asadelta
    model.compile(loss='mse',optimizer=RMSprop(),matrics=['mean_squared_error'])
    history - model. fit(x_train,Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data-(x_test,Y_test))
 
About how to handle big data with batch updating, watch vedio from Andrew Ng [mini-batch gradient descent](https://www.youtube.com/watch?v=-_4Zi8fCZO4)

Auto ensemble examples on Scikit, [feature importance plot](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py)

When you have very good financial features, DNN is not better than RF.  


    



    
    
