# Machine Learning and Quantitative Trading - Session 3

**Current Session Content**
> Financial Data Get and Save
> Time Series Data Analysis: Pair Trading Practise

**Selft pre-review knowledge**
> P-value
> T-statistics

# Part 1 - Data retrieve and Save

## Set up the DB meaning
 * Build Model (Good model $X$ will output good result $Y$)
     1. Model Set up
     2. Model analysis
     3. Model backtesting
     4. Risk control
* Trading
* Purpose: Save data in local or on the cloud, Read data from local or cloud.

## Financial Data Sourcing (Free)
* Financial Data of China - http://tushare.org/index.html
* Yahoo Finance - http://finance.yahoo.com
* Google Finance - https://www.google.com/finance
* QuantQuote - https://www.quantquote.com (S&P500 EOD data only)
* EODData - http://eoddata.com (requires registration)

**Also there is other way** to buy the high quality finanical data from the third parties.

## Financial Data Saving Method (Popular)
* CSV
* NoSQL
* SQL
* Hive
* MangoDB

## Financial Data Format (Necessary data to get online)
* Trading Organisation Name (eg. Nastaq, SP500, etc)
* Data source (eg. Yahoo Finance, Google Finance, etc)
* Ticker/Symbol (Stock symbol name on the market)
* Price
* Stock splits/dividend adjustments  (It may need adjust when download low quality finance data)
* National Holidays (Need remove from time series, it useful to good train your model)

**Data issues**, the stock splits may need recalculate according to formula if the raw data with low quality. Some noise data need be removed using spike filter. Repair absence data using Pandas to transfer as mean, mid or zero value.

## Self build MySQL (On Linux)
- Installation
  **sudo apt-get install mysql-server**
 - Type follow Linux command go to MySQL
 **mysql -u root -p**
  -  Following command to create user on MySQL
     mysql> CREATE USER 'sec_user'@'localhost' IDENTIFIED BY 'password';
     mysql> GRANT ALL PRIVILEGES ON securities_master.* TO  'sec_user'@'localhost';
     mysql> FLUSH PRIVILEGES;

## Design End of Day (EOD) table 
- Exchange
- DataVendor
- Symbol
- DailyPrice

**FYI** : It is general selection choice, it could be adapted by your preference. Or it may need adapt when using different API.


###  Create Exchange Table in MySqL
CREATE TABLE `exchange` (

    `id` int NOT NULL AUTO_INCREMENT,
    `abbrev` varchar (32) NOT NULL,
    `name` varchar (255) NOT NULL,
    `city` varchar (255) NULL,
    `country` varchar (255) NULL,
    `currency` varchar (64) NULL,
    `timezone_offset` time NULL,
    `created_date` datatime NOT NULL,
    `last_updated_date` datetime NOT NULL,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;

**Be careful**, using ` instead of ' in MySQL when creates a table, or without comma to introduce the keyword is okay.

###  Create DataVendor Table in MySqL
CREATE TABLE `date_vendor` (

    id int NOT NULL AUTO_INCREMENT,
    name varchar (64) NOT NULL,
    website_url varchar (255) NULL,
    support_email varchar (255) NULL,
    created_date datatime NOT NULL,
    last_updated_date datetime NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;

###  Create Symbol Table in MySqL
CREATE TABLE `symbol` (

    id int NOT NULL AUTO_INCREMENT,
    exchange_id int NULL,
    ticker varchar (32) NOT NULL,
    instrument varchar (64) NOT NULL,
    name varchar (255) NULL,
    sector varchar (255) NULL,
    currency varchar (32) NULL,
    last_updated_date datetime NOT NULL,
    PRIMARY KEY (id),
    KEY index_exchange_id(exchange_id)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;

###  Create DailyPrice Table in MySqL
CREATE TABLE `daily_price` (

    id int NOT NULL AUTO_INCREMENT,
    data_vendor_id int NOT NULL,
    symbol_id int NOT NULL,
    price_date datetime NOT NULL,
    created_date datetime NOT NULL,
    last_updated_date datetime NOT NULL,
    open_price decimal (19,4) NULL,
    high_price decimal (19,4) NULL,
    low_price decimal (19,4) NULL,
    close_price decimal (19,4) NULL,
    #adjustment close price
    adj_close_price decimal (19,4) NULL,
    volumn bigint NULL,
    PRIMARY KEY (id),
    KEY index_data_vendor_id (data_vendor_id),
    KEY index_symbol_id (symbol_id)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;

##  Connect to MySQL by python
`sudo apt -get install libmysqlclient -dev`
`pip install mysqlclient`

##  Task

If you have no idea for the symbol of the stock, can search in google, eg the list of [S&P 500 Companies]. Download the symbols from wiki to local DB, then according to the symbols to download related priced from Yahoo finance. 

Following are the example, how to get the symbol from wiki, then save those values to MySQL DB. According to the symbols in DB to retrieve its relevant price from Yahoo finance.  

- Step 1, 2: Use [insert symbol.py](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%203/insert_symbols.py) file to read and download symbols to local MySQL DB
- Step 3: Use [price retrieval.py](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%203/price_retrieval.py) to download the symbols price from Yahoo finance
- Step 4: Use [retrieving data.py](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%203/retrieving_data.py) to print out the data information, use it to compare with the data on Yahoo finance website, make sure they are consistently.

### Use pandas download Yahoo Finance Data
Python includes a package to directly download Yahoo finance data, if you want download other data and its API is not good, then you can use above steps to have the data.

Following is the code example to retrieve Yahoo finance data by using pandas.

    from __future__ import print_function
    import datetime
    import pandas.io.data as web
    
    if __name__ == "__main__":
        spy = web.DataReader(
               "SPY", "yahoo",
               datatime.datetime(2007,1,1),
               datatime.datetime(2020.10.15)
         )
         print(spy.tail())

## Download data from Quandl
The premier source for financial, economic, and alternative datasets, serving investment professionals. User [quandl data.py](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%203/quandl_data.py) to download its data. Those data is enough to setup model for quant researching. 

**FYI**: Quandl platform accepts 50 times/per day requesting without registration, with registration could have 500 times/per day requesting.

## Download data from tushare
The data source for Chinese financial, economic, if you are not familiar its API, you can find more information from [tushare data information](http://tushare.org/storing.html).

## Homework
### Assignment 1

### Assignment 2
Reterive the stocks from tushare, aggregate those stocks, lay out the different symbols on a 2D canvas.
**Hint**: http://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html#stock-market

# Part 2 - Time Series Data Analysis
Traditional theory for the time series, it has the stational sequence data, the finance product's price moves around with means, so when the price is higher than the mean value, we do short, lower than the mean vlue, we do long. Another theory is based on mometu.

In this section, we do time series analysis, to determinate it is random walk (the next price has no relationship with the previous price), if the price is random walk, then it could not be predicated, there is no way for arbitrage trading.  So the analysis purpose is to find some time series with patterns which does not random walk.

Most of the individual stock with random walk, but for the portfolios could be found linear relationship which with stational patterns, so could do time series analysis for portfolio to win profit.

- Means Reversion and Ornstein-Uhlenbeck process

     $dx_t = \theta(\mu-x_t)dt+\sigma$$dW_t$
     
     $\theta$ represents the speed of $x_t$ return back to mean (time series) value $\mu$, $\sigma$ represents variance (time series) , $W_t$ is Brownian motion. 
     
     **This formula represents** the wave of the price $dx_t$ has proportional to $(\mu-x_t)$ plus random noise.

####  How to test the input is stable time series  or not?
Suppose giving time series $x_1, x_2, x_3,.....x_t$ .  
 **Two possible ways to test:**
 1)  ADF Test -> Unit root in autoregressive time series
    $\Delta y_t = a + \beta t + \gamma y_{t-1} + \delta_1\Delta y_{t-1} + .... + \delta_{p-1}\Delta y_{t-p+1} + \epsilon_t$         
   * Calculate the $test \space statistic, DF_\tau$, which is used in the decision to reject the null hypothesis
   * Use the $distribution$ of the test statistic (calculted by Dickey and Fuller), along with the critical values, in order to decide whether to reject the null hypothesis
   => If the $\gamma$ value equal to 0, which means the current price is not relevant to previous price.

The stats calculation has been include in Python model, can using **Pip install Statsmodels** to download, then following is the example code how to use it.

    # Download the Amazon OHLCV data from 1/1/2000 to 1/1/2020
    amzn = web.DataReader ("AMZN", "yahoo", datetime(2000,1,1), datetime (2020,1,1))
    # Output the results of the Augmented Dickey-Fuller test for Amazon
    # with a lag order value of 1
    ts.adfuller(amzn['Adj Close'], 1
Output result is like

    ( 0.049177575166452235,
      0.96241494632563063,
      1,
      3771,
      { '1%' :  -3.4320852842548395,
        '10%' : -2.5671781529820348,
        '5%' : -2.8623067530084247 } ,
      19576.116041473877 )
Analysis the output result:
Compare the first line value with the all the values in {}, eg the value of 1%, 10%, 5%, the first time value is bigger than all of them, means it has no hypothesis. So the $\gamma$ value will be 0,  means the stock price of Amazon is random walk, we could not predict it. 
After analysis several individual stock, you will find most of them are following random walk, but we still have chance to win the market, that is **portfolio investment**.
 
 2)  Hurst Exponent
 Compare with the 1st method, the 2nd method is more simple to determinate the time series is stable or not.
 $<|log(t+\tau) - log(t)|^2> ~ \tau^{2H}$
 - $H$ < 0.5 - The time series is mean reverting
 - $H$ = 0.5 - The time series is a Geometric Brownian Motion
 - $H$ > 0.5 - The time series is trending

Here is the simple implemention of this method, using the returned $H$ value compare with 0.5.

    def hurst (ts) :
           '''Returns the Hurst Exponent of the time series vector ts'''
           # Create the range of lag values
           lags = range (2, 100)
           # Calculate the array of the variances of the lagged differences
           tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
           # User a linear fit to estimate the Hurst Exponent
           poly = polyfit (log(lags), log(tau), 1)
           # Return the Hurst exponent from the polyfit output
           return poly[0]*2.0
Then according to the time series definition for mean reverting, Geometric Brownian Motion and Trending, can create relevant time series.

    # Create a Gometric Brownian Motion, Mean-Reverting and Trending Series
    gbm = log(cumsum(randn(100000))+1000)
    mr = log(randn(100000)+1000)
    tr = log(cumsum(randn(100000)+1)+1000)
    # Output the Hurst Exponent for each of the above series
    # and the price of Amazon (the Adjusted Close price) for
    # the ADF test given above in the article
    print("Hurst(GBM): %s" % hurst(gbm))
    print("Hurst(MR): %s" % hurst(mr))
    print("Hurst(TR): %s" % hurst(tr))
    # Assuming you have run the above code to obtain 'amzn'!
    print("Hurst(AMZN): %S" % hurst(amzn['Adj Close']))

Output result is

    Hurst(GBM): 0.502051910931
    Hurst(MR): 0.000166110248967
    Hurst(TR): 0.957701001252 
    Hurst(AMZN): 0.454337476553

So the Hurst Exponent has the same conclusion with 1st method, the individul stock price follows Geometric Brownian Motion, it is random walk.

## Homework
### Assignment 3
Using above two methods to analsyis the stocks in S&P 300, report which individual stock has mean-reverting (if exists)

## From Individual Stock to Portfolio (Pair Trading)
We have analyzed, most of individual stock are random walk, the price could not be predicted, the portfolio could find the pattern to win the profit.

* Cointegrated Augmented Dickey-Fuller Test
      Suppose $y(t)$ and $x(t)$ has the similiar wave trending, then we have model:
            $y(t) = \beta x(t) + \epsilon(t)$
            At time t, the price of stock $y$ has a constant multiple $\beta$ of the price of stock $x$ plus random noise $\epsilon$ . 
           **Question**: How to gurantee the noise $\epsilon$ (residual) has Gaussian distribution (~$N()$) ?
     
     An example [cardf.py](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%203/cadf.py) plot two stocks wave trending to see whether they have linear relationship. Base on the output results of Residual plot, then do Augmented Dickey-Fuller test, the output result is like:
     
         ( -2.9607012342275936,
      0.038730981052330332,
      0,
      249,
      { '1%' :  -3.4568881317725864,
        '10%' : -2.5729936189738876,
        '5%' : -2.8732185133016057 } ,
      601.96849256295991 )

Analysis the output result:
Compare the first line value with the value of 1%, 10% and 5%, the first time value is smaller than the value of 5%, means it has hypothesis. There is stable time series at 5% range for portfolio ($x \space and \space y$). 

**Pay attention**: You may find the range of time series are stable, but it does not have generalization ability, so it is a good strategy for trading.

## Homework
### Assignment 4
Retrieve the stocks from S&P300 to find pair of stocks has **cointegration relationship**, then print them out. 

# Machine Learning: A different Approach
- $Y=f(x)+\epsilon$, $x$ could be in high dimension, eg $\bar x_{500, t} (f_1, f_2...f_N)_t$ with 500 dimensions and $N$ features at time $t$.
- Logistic regression (classification)
- SVM (classification or regression)
- Random Forest (classification or regression)
- LSTM (regression or classification), it is variant of RNN, it has better memory than HMM.
- .....and more

Compare Lasso and OLS for features selection, Lasso is better. It has the mechanism to find the better features if you did not have optimal features to be used. If you already have the lag term in the data, better not use LSTM, it is the model to help find the lag term if it does not in dataset. SVM is help to find the biggest seperate boundary, or do outlier detection.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen.)

   [Learn Python The Hard Way]: <https://learntocodetogether.com/learn-python-the-hard-way-free-ebook-download/>
   [S&P 500 Companies]: <https://en.wikipedia.org/wiki/List_of_S%26P_500_companies>


