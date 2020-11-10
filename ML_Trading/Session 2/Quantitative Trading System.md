# Machine Learning and Quantitative Trading - Session 2

  - Basic Financial Knowledge Introduction
  - Basic Python Knowledge Introduction

**Current Session Content**
> Learning Python and its data processing libraries
> Using machine Learning to do technical analysis

**Next Session Content**
> Using Python crawling financial data
> Using Python processing financial data: cleaning data, data visalization, feature extraction, etc
> Build a quant strategy model base on machine learning

## Pre-installed
* Anaconda installed
* ipython notebook

## Python basic knowledge
* Python Type : str, float, bool, int, long
* Python basic gramma : branch, loop, function
* Python data structure : tuple, list, dictionary, etc
* Python build-in function
* Python and objective programning

More details can learn [Python] in Y minutes.

### Numpy basic knowledge
* Using numpy do linear operation
  1. Create matrix, vector, etc
  2. Master on matrix index
* Numpy input and output
* Numpy basic functions

Self book reading from chapter 4 of [Python for Data Analysis]

### Pandas basic knowledge
Pandas library could provide more efficient computation way than numpy, following knowledge in Pandas is mandatory to know.
* Pandas and data io
* Dataframe related functions (eg. statistical index, draw up)
* Index of Pandas

Self book reading from chapter 5 of [Python for Data Analysis]

### Sklearn basic knowledge
It includes most all traditional machine learning models, except deep and reinforcement learning model.
* Using Sklearn do classfication on mnist dataset
* Using Sklearn do linear Regression (LR) model

Self learning for [sklearn].

## Question for basic technical issues
Most of the developing issues you could find answer in [stackoverflow], try solve your questions using google first, it is also a skill as quant.

## CAPM Model
* Asset Portfolio [a\%, b\%, c\%]
  |(a\%)| + |(b\%)| + |(c\%)| = 100\%
* Market Portfolio
  1. SP500
  2. DOW J
  3. etc
* Individual Stock's CAPM model (It is a linear model)
  //At t time, the individual stock returns equal to, at the t time,  the whole market return * value of the beta + its individual value of alpha
  $r_i(t)$ = $\displaystyle \beta_i$ * $r_m(t)$ + $\displaystyle \alpha_i(t)$
  //Theoretical financial believes the Alpha residual will be $0$, like $\displaystyle \sum^{+\infty}_{-\infty}$ $d_i(t)$ dt = $0$
  If CAPM says E $(\displaystyle \alpha(t))$ = $0$
 The CAPM model told us, the individual stock and stock market with linear relationship, like, $r_i(t)$ as y = f(x) + $\displaystyle \epsilon$ (noise), the noise $\displaystyle \epsilon$ be $0$. So we want to know what kind of reasons (inputs x) affect to the return of stock (y) at t time (negative or positive return). So in the valid market (CAPM model think), the individual stock's return is (only) affected by the real market, it is linearly. Every stock has its pair of alpha and beta value. But our real market is far from the valid market, this is why we can get profit/earn money from current market.
 
 Using machine learning technicals to analaysis/learn from history data (big data) to know the beta value.
 
* Some Grahpical interpretations
  **CAPM Model**
![CAPM](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/SCL-plot.png)
Using SVM model linearly learn the history data, the linear line is the CAPM model, but you will see all the history data could not be perfectly seperated by linear line.

* Passively managed funds and actively managed funds
-- Passively managed funds: can't beat the market, can't find the optimal beta value to get arbitrage, just copy the market index, hold
-- Actively managed funds: can find the optimal beta value to get arbitrage, so choose individual stocks, trade frequently
* $r_i(t)$ = $\beta_i$ * $r_m(t)$ + $\alpha_i(t)$
**Key point**: in above formular, passively managed funds think that the alpha is random noise, the exepctation of alpha is zero, in $r_m(t)$ already includes all the market information, it is hopeless to get profit from the market. Actively managed funds think the $\displaystyle \alpha_i(t)$ with the meaning, through disassebling it, can find more variables to get arbitrage. This means, in machine learning function f(x), it not just with variable rm, it will also include other varables, eg. MA5, MACD, etc.

## CAPM portfolio model
$\begin{aligned}
r_p(t) & =\sum_i W_i \cdot P_i(t) \\
& = \sum_i W_i (\beta_i r_m(t)+\alpha_i(t)) \\
& = \sum_i[W_i\beta_ir_m(t)+W_i\alpha_i(t)] \\
&=\sum_iW_i\beta_ir_m(t) + \sum_iW_i\alpha_i(t) \\
r_p(t) &=\beta_pr_m(t) + \begin{cases}\alpha_p(t)\\ - \end{cases}\\
\end{aligned}$

In case we have three stocks in the portfolio, at time t, the return of portfolio is $r_p(t)$, at right side, expand the $P_i(t)$ in CAPM model, then we add the weight value $W_i$ into CAPM model, get CAPM portfolio model. Then we transfer $W_i\beta_i$ to be $\beta_p$, the beta of a portfolio is the weighted average of the assets that make up the portfolio, there is linear relationship between the portfolio and the market.

  **CAPM Portfolio Model**
![CAPM Portfolio](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/CAPM%20Portfolio%20Model.gif)

## Some inference
E($\alpha$) = $0$
The $\alpha$ value following normal distribution, pick up sample [$\alpha_1$, $\alpha_2$, $\alpha_3$, $\cdots$, $\alpha_n$] from $\alpha$, E($\alpha$) = $0$ means $\frac{1}{n}$ $\sum\alpha_i$->$0$, n->$\infty$. So base on CAPM model, $r_i = \beta_ir_m + \alpha_i$, in bull market, the $r_m$ is $\Uparrow$, we try maximal $\beta_i$ to maximal the profit of $r_i$, in bear market, the $r_m$ is $\Downarrow$, we try minimal $\beta_i$ to mimial the lost of $r_i$, it is based on linear theory between $r_i$ and $r_m$.
**So we have:**
Bull market: choose $\beta$ larger than $r_m$
Bear market: choose $\beta$ smaller than $r_m$, negative value is allowed

**Again, if the efficient market hypothesis is valid, then no one could predict the financial market, no proper $\beta$ value could be found from history data.**

The quantitative trading system is based on inefficient market hypothese, analysis the $\alpha$ and $\beta$ value in CAPM model to find the arbitrage mechanism, do example for one of CAPM derivatives, $r_i = \beta_ir_m+r_iMA5$ which means the $r_i$ is impacted by $r_m$ and index MA5.

## Arbitrage price theory (APT)
$r_i(t) = \beta_i * r_m(t) + \alpha_i(t)$

In CAPM model, the $\beta_i$ is constant value with market $r_m(t)$ at t time, in APT, the $\beta_i$ is variable, it represents several different segments in the market $r_m(t)$, eg $\beta$ = w*r, w repsent the $\beta$ of individual segment, r represents its relevant market segment. Then we will have the linear regression model between the return of individual stock and the return of each segment market.

Example for portfolio with two stocks
- Stock A: +1% mkt, $\beta_a = 1.0$
- Stock B: -1% mkt, $\beta_b = 2.0$
 
So we do long A, short B. Through history data, the machine learning technical could find the optimal $w$ value for stock A and B to get maximal return of $r_p(t)$

## Technical Analysis vs Fundamental analysis
History Data:
- Price, trading volume
- Features (market index)
- Heuristic choice (Experience, Machine Learning)

Through the history data, using machine learning technical to find which function $f$ is suitable for those data, and the input value of $x$ (features) could get best output $y$, $f(x) = y$. The data mining technical to do data extracting, get indicator/features $\bar{x}$ from history data, different input $\vec{x}$ will affect to the model prediction accuracy. Normally the technical analysis will base on price, trading volumn to mine the features, then do heuristic features choice base on trading experience or machine learning technical for automatic selecting, eg, using genetic algorithm (GA) automatically pick up the best features to build the trading model.

### When technical analysis works
- Multiple indicators/features with non-linear combination
- Short time (Micro-scale prediction, suitable for day, hour or minutes trading)
- Heterogeneous monitoring (Find inconsistency between individual stock and market for arbitrage trading)

### Trading time horizen
- Technical trading time horizen
The model prediction accuracy declines over time.
- Fundamental trading time horizen
The prediction analysis accuracy rises over time.

## Fundmental technical indicators for machine learning
- Momentum
mom[t] = price[t] / (price[t-n]) - 1
- SMA: Simple Moving Average. (smooth, lagged)
Consider it as a smoothed market price, or you can consider it as a filter to remove the noise which helpful analysis in machine learning, 
- BB (bollinger bands) BOLL index
 It bases on SMA value to calcuate two decision boundary value, it is a std (standard deviations) of time series. Because the SMA could have big difference between true and market price in higher amount frequence. It can guarantee the trading strategy is robust.
- Normalization
Norm = (value - mean)/values.std(), 
It is another index to help machine learning analysis. Consider there is a function $y = f$(SMA, MOM, BB), the $\bar{x}$ with SMA [-0.5, +0.5], MOM[-0.5, +0.5], BB[-1, +1], other index $x$ could be in range [-10, 35], if we don't do normalization, the index range with bigger value, it will impact the output result $y$ more than other indexs which with smaller ranger. Eg, $y = ax_1+bx_2+cx_3$, if the $x_3$ is huge big, then $y$ will be directly impacted by it, other variables $x_1$ and $x_2$ are ignored. So doing normalization which involves all kinds of indexs to impact on the output reuslt $y$. 

### An example regarding to linear regression line
![Linearline](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/Linear%20regressions%20of%20Sales%20vs.%20TV%2C%20Radio%2C%20Newspaper.png)
Above figure shown **Sales** vs **TV**, **Radio** and **Newspaper**, with a blue linear regression line fit seperately to each. If we consider the Sales can be predicted using TV, Radio and Newspaper, then we can build a model, $Sales \approx f(TV, Radio, Newspaper)$, here **Sales** is a response or target hat we wish to predict. We generically refer to the response as $y$.
**TV** is a feature, or input, or predictor, we name it $X_1$. Likewise name **Radio** as $X_2$, and so on. We can refer to the input vector collectively as

$\bar{x}_3 = \left(\begin{matrix}X_1 \\X_2 \\ X_3 \end{matrix}\right)$

Now we write the model as

$Y = f(X) + \epsilon$

where $\epsilon$ captures measurement errors and other discrepancies. It incase the $f(X)$ is not prefectly, actually all the functions we find (actual $f$) will have discrepancies with expected $f$, it is the general machine learning (supervised learning) model to find $E(\epsilon) = 0$.
- With a good $f$ we can make predictions of $Y$ at new points $X=x$.
- We can understand which components of $X = (X_1, X_2,..., X_p)$ are important in explaining $Y$, and which are irrelevant. eg. **Seniority** and **Years of Education** have a big impact on **Income**, but **Marital Status** typically does not.
- Depending on the complexity of $f$, we may be able to understand how each component $X_j$ of X affects $Y$. 

### An example regarding to regression function
The simplese model is KNN (k-nearest neighbors algorithm), suppose we have history data, we have the model $f$ without defined input $\bar{x}$, but base on history data, if there is a new input value $x^*$, according to the history data distribtuion, we know what the present value $y$ for the new input value $x^*$ could be. eg. new input value $x^* = 5.7$, check in history data, we could have
|$x$|$y$|
|---|---|
|5.7|1.2|
|5.7|3.6|
|5.7|2.0|

Base on above figure, there is a value of x with three different value of y, so we do $E(y|x=5.7)$ means **expected value** (average) of $y$ given $x = 5.7$, so the new input value $x^* = 5.7$ has output $\hat{ y}$, if the new input value $x^* = 5.732$, it does't exist in history data, then choose the closest $x$ value, like $x=5.7$ to get the $y$ value. This is the KNN algorithm core

Here is an example:
![regression function](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/Regression%20function.PNG)

Base on above figure, whether there is an ideal $f(x)?$ In particular, what is a good value for $f(x)$ at any selected value of $X$, say $X=4?$ There can be many $y$ values at $x=4$. A good value is 
$f(4)=E(Y|X=4)$

This idea $f(x)=E(Y|X=x)$ is called the **regression function**.
- Typically we have few if any data points with $X=4$ exactly.
- So we cannot computer $E(Y|X=x)!$
- Relax the definition and let 
$\hat{f}(x) = Ave(Y|X \: \epsilon \:N(x))$
where $N(x)$ is some **neighborhood** of $x$.
![KNN neighbor](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/KNN%20neighborhood.PNG)
KNN has dimension problems, in one dimension, it without problem, when the dim(x) > 10, the KNN model performance will be decreased sharply, it caused by dimensional disaster problem, searching KNN in high dimension, it nearly search in entire high dimension space, so analysis data with high dimension, we will use other more complex machine learning model. 

### An example for linear model with high dimension
![Linear Activation](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/linear.png)
The **linear** model is an important example of a parametric model:

$f_L(X) = \beta_0 +\beta_1X_1+\beta_2X_2+...+\beta_pX_p$
- A linear model is specified in terms of $p+1$ parameters $\beta_0,\beta_1,...,\beta_p.$
- We estimate esitmate the parameters by fitting the model to training data.
- Although it is **almost never correct**, a linear model often serves as a good and interpretable approximation to the unknown true function $f(x)$

If $f_L(x)$ is current stock price, $X_1$ as MA5, $X_2$ as MA10, $X_3$ as MACD, this linear model can present the stock price is linearly with the market indicator. 
![ReLU Activation](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/ReLU.png)
Actually, deep learning also could be considered as linear model, because most of DL models are using ReLU activation, it is non-linear transformation which very close to linear transformation, there is an interesting research paper [Fooling CNN] talking about it. So, linear model also could good handle high dimension data, not only non-linear model.

### An example for linear vs quadratic model 
There are multiple linear/regression lines can be set up to sepereate the data, the line with minimize **MSE** value will be selected.
![Multiple regression line](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/multiple%20regression%20line.png)
MSE (Mean square error)  calculates the residual for every data point, taking squares value of each, after that summing them all, then take the average of all these residuals.
![MSE](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/MSE.jpg)
$MSE = \frac{1}{n}\sum(y-\hat{y})^2$
$\frac{1}{n}\sum$ is the input (test) dataset, $y$ is predict value, $\hat{y}$ is actual value, $(y-\hat{y})$ is the difference between actual and predicted. The machine learning model will automatically learned a regression line which with the smallest MSE value, using this linear line to seperate the data. 
![linear vs quadratic](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/Basic%20linear%20and%20quadratic%20model.png)
Compare with a linear model $\hat{f}_L(X) = \hat{\beta_0} + \hat{\beta_1X}$, it gives the reasonable fit is slightly worse than a quadratic model $\hat{f}_Q(X) = \hat{\beta_0} + \hat{\beta_1X} +\hat{\beta_1X^2}$. So, what kind of model is better? It is based on your knowledge of the data, it could be linear or quadratic, the best way to validate and select a model, it need some testing data to validate, those data never be used for model training. Finally, we choose the model which with the smallest MSE value.

### An example for non-linear and liner model in 3D
There are two predictors used to estimate income. 
Years of education and seniority are fit by a non-linear 3-dimensional surface
![Non-liear fit](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/Non-linear%20fit%20for%20income.png)

$income = f(education, seniority) + \epsilon$ 
Red points are simulated values for **income** from the model, $f$ is the blue surface.

Years of education and seniority are fit by a linear 3-dimensional surface
![Linear fit](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/Linear%20fit%20for%20income.png)

$\hat{f_L}(education, seniority) = \hat{\beta_0}+\hat{\beta_1}\times education + \hat{beta_2}\times seniority$
Red points are simulated values for income from the model, $\hat{f}$ is the yellow surface.

Even more, if we have larege number of observations, we can not make any assumptions about the form of $f$, but try to estimate $f$ by getting as close to the data point, we can call it is spline regression model.
![Spline regression fit](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/Non-linear-spline%20fit.png)

The model $\hat{f_s}(education, seniority$ fits to the red points without errors. In machine learning, it also known as **overfitting**.

## The Trade-Off Between linear and non-linear model
Normally a model with higher flexibility (complexity), the learned $f$ will be in non-linear and it will have less interpretability, eg. SVM, NN. A model with lower flexibility (complexity), the learned $f$ will be in linear and easy to understand, eg. Subset Selection, Lasso.
![Model complexity](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%202/IMG/Model%20complexity.png)

Suppose we have fit a model $\hat{f}(x)$ to some training data $T_r$, and let $(x_0, y_0)$ be a test observation drawn from the population. If the true model is $Y=f(X)+\epsilon$ (with $f(x)=E(Y|X=x)$), then 

$E(y_0-\hat{f}(x_0))^2 = Var(\hat{f}(x_0))+[Bias(\hat{f}(x_0))]^2 + Var(\epsilon)$

The expection averages over the variability of $y_0$ as well as the variability in $T_r.$ Note that $Bias(\hat{f}(x_0)) = E[\hat{f}(x_0)]-f(x_0).$ Typically as the **flexibility** of $\hat{f}$ increases, its variance increases, and its bias decreased. So choosing the flexibility based on average test error amounts to a **bias-variance trade-off**.

Explained above phenomenon in machine learning, it means when the model fitting the (training) data pefectly, maybe in overfitting, the bias value will be in small, but the variance will be in big, because the model will not good fit for "new" (test) data,  the spline regression model may have small Bias, big Variance value. So, if the model fits on training data with some errors, then it fits on test data is better than training data, then the model has been good learned. The **key point in machine learning** is, find the **optimal combination of bias and variance** to generate the best model.

# Reference for self-study
Classical statistical book: [The Elements of Statistical Learning]
Course of Introduction To Statistical Learning: [Introduction To Statistical Learning]

Next Session
---
### Part III

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen.)

   [Python]:<https://learnxinyminutes.com/docs/python/>
   [Fooling CNN]: <https://github.com/rmrisforbidden/Fooling_Neural_Network-Interpretations>
   [The Elements of Statistical Learning]: <https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf>
   [Introduction To Statistical Learning]: <http://www.rnfc.org/courses/isl/>
   [Python for Data Analysis]:<https://www.programmer-books.com/wp-content/uploads/2019/04/Python-for-Data-Analysis-2nd-Edition.pdf>
   [sklearn]:<https://scikit-learn.org/stable/>
