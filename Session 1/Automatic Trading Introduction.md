# Machine Learning and Quantitative Trading

  1. Why need algorithm trading
  2. Trading system develop and design
  3. Time series analysis
  4. Strategy modelling and optimize
  5. Strategy evaluation and backtracking
  6. Risk management
  7. Trading strategy implementation
  8. Trading strategy execution

The pre-condition required:
  - Scripting knowledge (will be in Python)
  - Basic financial trading knowledge
  - Machine learning / Deep Learning

## Quant Trading Introduction
Work as quant, the important capability is discovering patterns in the data, but the data generates from the trading market will huge, at least dozens GB size of data everyday, manual analysis nearly impossible. So, we need machine learning, using this technology to build the mathmatical modelling  which is based on learning from bid data, to find the trading logical behind, then setting series of rules to simulate the trading behaviour. 

### Advantage
1. History data evaluation (model driven by data)
2. Execution effectively
3. Without irrelevant emotion input (if you know the win rate, then you do thought times then you will always get profit)
4. Can be scalled evaluation (Scalled by scientific method)
5. Can do high frequency trading

### Disadvantage
1. Cost (time and cash, eg for stock index futures, it need big amount money) 
2. Technical (strong background in math and coding)

## Quant Trading procedure
Pre-condition (base on open source or self-building platform)
1. Proposal hypothesis
2. Build trading model (without GPU for deep learning nearly impossible)
3. Backtracking validation
4. Exeuction trading

## Quant Trading strategy sourcing
1. Market micro structure research (for HFT mostly which with a lot of noise and need analysis deep information, eg base on momentum, limit order books)
2. Fund structure arbitrage (some investment company with behaviour pattern, can base on it to get arbitrage)
3. **Machine learning / Artificial intelligent** (can be used by independent person base on history data)

### Order Book for arbitrage
| Bid/Buy | Price | Amount
| ------ | ------ | ------ |
| Bid 1 | 100 | 1000
| Bid 2| 101 | 500
| Buy 1 | 99.95 | 50
| Buy 2 | 98 | 150

The micro structure will base on the order book to forcast the next trading prices (in seconds)

### Keypoint of machine learning
The parameters setting in the modelling of machine learning, should better understanding why choose this machine learning model and its relevant attributes. So now days, we are using unsupervised machine learning to learn the optimal parameter settings for the modelling.The familair model we know, eg SVM, DT, DNN, LR, NB, RF, etc. Different data input, there is suitable model, the suitable classification in the model can properly seperate the data according to its features. So select the features are also very important. Later we will discuss what kind of models are suitable for financial data.

### Four paradigms of machine learning
- Connectionism (It is learned with higher accurate, but hard to be explained, eg. NN)
- Symbolism (Its expression can be simply and clearly explained, some of them are used in financial, eg,DT)
- Frequentists
- Bayesian

## A (super) Brief History of A.I
- 1958 to 1960 (NNs, Logistic)
- 1970's (A.I Winter period)
- 1980 to 1995 (NN)
- 1995 to 2010 (SVMs and statistical learning) 
- 2010's toc current (DBN, Alexnet, DNN, RNN)

** NN is strong for feature expression, it is good use for image, because human does not know how to good express the image, but financial does not need extra features learning. Logistic model hard to analysis big data.

## Machine Learning in a nutshell
- Data (it is the driven for the ML)
- Model&Objective Function
- Optimization (search extreme point for the objective function)

# Machine Learning and Trading
Using ML for trading, could be implemented in following:
- Limited Order Book Modeling (Micro structure, forcasting in seconds with very high data noises)
- Price-based Classification Models (Input X bases on market price index,eg. MA5, MACD, rolling average, open price, close price,etc, the output y do the classification, eg. Up, Down, Even)
- Text-based Classification Models (Base on news, sociatl network, eg. twitters which using NLP to select the stocks, input the text as x, then output the toppest 5 stocks as y, **using it on risk management will be better**)
- Reinforcement Learning (It is difference with previous supervised ML, another model, eg. Alpha Go)

Normally in privacy equity, the trading could be used:
1. Pure scripting  (rewrite all function libraries to increase trading speed, without ML knowledge)
2. Base on open source platform (based on experiments to create the trading strategy, normally we call them as quant)

**In the future, more and more financial market using the trading strategies, all the goods will be quickly returned back to its reasonable price whatever it is overestimated or underestimated.

## Four Key Factors that makes magic happens
- Good Model and Efficient Training Algorithms (Deep learning found the efficient training algorithms since 2006, it find a way to set the optimal value for W)
- Hardware (GPU for ML and DL / CPU only for ML)
- Data (High Quality, less noise and could be explained)
- Platform (Keras and Tensorflow for DP, Sklearn for ML, Python is the platform with all the libraries to reuse the ML, DL models)

## A little aside of deep learning
Deep learning for the NN which with more than two hidden layers
- CNN for spatial data
- LSTM for time series data (A type of RNN)

**LTSM is better than HMM (Hidden Markov Model), LSTM could handle big data amount**

## A little aside of reinforcement learning
It supports there is an environment, an agent, the agent will select an action from its policy libraries, eg. what kind of situation buy the stock, sold the stock. Every moment the agent gives an action base on current environment, the environment will base on the action input, feedback a reward to the agent, finially, the model (reinforcement learning) to find a policy which has **maximum reward** in limited times. It using the history data to simulate trading stocks thousand times, then the model will learn a proper policy, after that you can use this model to run for real stock market. **It is an efficient way to build the model, super cool, but need high level ML skills**. 

**Recommend online course:** 
Stanford Reinforcement Learning - [CS234]

## A little aside of natural language processing (NLP)
Normally it uses word2vec technical, represent the text input to be vector. NN can use this technical to do market implementation. The text input to Encoder (need ML technical), then output vector x as representation, it input into model M to do classification, output as y. It presents the word meaning is related to its next word, so it is a light NN, only includes one hidden layer.

### Key problem
How to define the input features?
- Feature engineering (according to financial experience/knowledge which you think important or necessary to be selected as the indicator, it could be more than 200 features)
- Feature selection (using data mining technical to decrease the features dimension, eg. using PCA)

### Example for stock forcasting
| Time | Name of Stock (Features)                   
| ---- | ---- 
|0001 | [Feature 1, Feature 2...Feature 8]
|0002| [Feature 1, Feature 2...Feature 8]

|Time|Y (Price)
|---|---
|0001|25
|0002|25.3

Base on above table, the input x could be time 0001 with 8 featues, the output could be the Y value in time 0002, the model learning bases on the history data mapping (x and y), could predict the next time series output y value. How to correctly define the features value in vector x is also important.

**Attention: Don't put the features' value which taking from future time in the history data.**

# Quant Trading Evaluation
1. Strategy basic hypothesis (eg. momentu, mean reversion)
2. Sharp Ratio
3. Levelage
4. Frequency
5. Risk
6. W/L
7. Model complexity (eg. NN, LR with different complexity, its estimation base on VC dimension)
8. Maximum drawdown
9. Bencharking

# Backtesting
Using the history financial data to validate the trading strategy. 

## Backtesting meaningful
1. Strategy screening (pick up the best result strategy)
2. Strategy optimisation (super parameter need train and learn from data)
3. Strategy validation (test it before use the strategy in real market)

## Negative backtesting method
Most of bias result in testing and real market is caused by:
1. Optimism bias (special time period with good result, the training data is selected for some purpose)
2. Time lagging (current data selecting with some future data features, it makes the results have the illusion of higher accuracy)
-> Program Bug
-> Train/Validation/Test set
3. Survivor bias (test data selection from history only include better performances stock, eg. only select the stock company which is survived from stock market crashing)

# Tools and language
Python
- Sklearn
- Pandas
- And more...

# Quant as project (Self scripting)
It  will be driven by event, many trading system, eg. Quantopian, Uqer, etc other open source platform they are sharing the similar scripting, the keypoint inside of the script is **"event driven"**, when the trading market openning, the event driven will forever looping until all events are processed:

### #event driven
```
While True:
    new_event = get_new_event ()
    if new_event.something == "whatever"
       do_something()
    if new_event.something == "all right"
       do_something_else()
    #wait 50 milliseconds (it is listener time setting, eg. HFT or HLT for higher or lower frequence trading)
    tick(50)
 ```
 
## Following model functions are necessary in the project:
- Event
- Event Queue
- DataHandler
- Strategy
- Portfolio
- ExecutionHandler
- Backtest

Next Session
---
### Part II

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen.)


   [CS234]: <https://www.youtube.com/watch?v=FgzM3zpZ55o&list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
