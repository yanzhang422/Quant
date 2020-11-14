# Machine Learning and Quantitative Trading - Session 4

**Current Session Content**
> From OLS to kernel machines and beyond
 - OLS (Ordinary Least Square)
 - Ridge
 - Lasso
 - Kernels (Transfer linear to non-linear method)
 - Cross-validation
 - Hands on: sklearn
 
## What is Machine Learning?
![ML_IMG](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/What%20is%20machine%20learning.PNG)
Difference between the traditional programming and Machine Learning, the actions trigged by rules, those rules are not defined by programmer, they are learned by models base on the patterns in the training data. The advantage in ML, the rules be mined may better than defined by humanity, because the machine could handle massive data from high dimensions.

Examples of Machine Learning
![Examples](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/machine%20learning%20examples.jpg)

3 Types of Machine Learning
![Three Type of ML](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/Three%20type%20of%20ML.jpg)
The critical method in Reinformcement is MDP (Markov Decision Process), it uses massive data to learn the parameters for MDP model.
### Supervised Learning
![supervised learning](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/supervised%20learning.png)
Supervised learning with two models, classification and regression.  The classification model will project the new input data, base on its position close to which class, then output its class name, eg short or long, down or up. The regression mode will output the predicted price at time $t+1$ according to the existing data's price when there is new data input at time $t$. 
### Unsupervised Learning
![unsupervised learning](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/unsupervised%20learning.PNG)
At high dimensions space, eg, the point $x$ has 300 dimensions * 1,  $x_{300} =[P_{t-1}, P_{t-2}, MD_5, MACD, ....total \space 300 \space indicators]$. So if took ten million records, then will have feature vector size is $[1000 (ten \space thousand) * 300]$. Compression means dimensionality reduction, eg PCA, because the features which created initially, it could be with redundant information, the compression could help keep the meaningful features, remove unnecessary features. In math, we call this is **"flow pattern"**, using the point in low dimension represents the point in high dimension (low dimension embedding).

In general, we using unsupervised learning (compression) to process the raw data, then using supervised learning model to output the result, it could increase the data quality and decrease model building time. Because the model complexity is related to ploynomial level of features. 
### What is Sklearn?
![sklearn](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/sklearn-cheatsheet.png)
Sklearn includes all the packages for machine learning except some packages for deep learning. When you build the model, most of code are handling data processing and visualization, by call Sklearn package in two or three code lines can finish a model building.
**Attention**: For dimensionality reduction, if the size of input data is  less than < 30k, using PCA decreases to 50 dimensions, then using TSNE decreases to 2 dimensions or customize to the dimensions you want. Because **TSNE is the best** one in all dimensionality reduction methods, but it has very slow running speed.
#### The simplest Sklearn workflow
    ##Retrieve input raw data
    train_x, train_y, test_x, test_y = getData()
    ##Build ML model with setting of parameters' value 
    model = somemodel()
    ##Training the model according to the setting values to find the best optimal combination for the input data
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    ##Compare the predicted result with the expected result (test_y) to score the model performance
    score = score_function(test_y, predictions)

**Attention**: most of time, the ML tells us what could not do, did not tell us what could do, so sometimes the model with high performance score, the result (backtest) also could be not good.
#### Flower Classification
![sklearn iris flower](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/sklearn%20iris%20flower.png)
Iris flower is the basic dataset in sklearn,  it has totally 150 kinds of flowers (records), each flower (record) represents by 4 dimensions $\bar X = [Sepal \space length, Sepal \space width, Petal \space length, Petal \space width]$, and three classification (output) are Setosa, Versicolor, Virginica.

Simple code

    from sklearn.datasets import load_iris
    iris = load_iris()
    ##The resulting dataset is a bunch object, you can see what's available using the method kyes():
    iris.keys()
 output: dict_keys ([ 'data', 'target', 'target_names',  'DESCR', 'feature_names'])
 Problem for iris dataset, its with small size, only has 150 records, for statistical analysis is enough, but for machine learning analysis which is too small.
 #### Digits Classification
 ![sklearn digits](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/sklearn%20digits.png)
 It is a classifical dataset (MNIST) for ML analysis, it has 60 thousand records and 28*28 vector features. In sklearn dataset, it was decreased to 8*8 vector features, keep in same records (60 thousand).
#### Generating Synthetic Data
![sklearn make a curve](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/sklearn%20make%20a%20curve.jpg)

 Sklearn has a function to automatically synthetic data into classification or regression. 
Simple code

    from sklearn.datasets import make_classification 
    # or
    from sklearn.datasets import make_regression

## Regression Shrinkage and Selection via the Lasso
### Regularization
Since the linear regression is the training matrix data $X_{N*d}$ multiple new vector $\theta_{d*1}$, return the new vector $Y_{N*1}.$

$$Y_{N*1} = X_{N*d}\theta_{d*1}$$

We can update the form to be
$$\bar Y = \bar X \bar\theta$$

All the answers so far are of the form if X and Y is linearly

$$\hat\theta = (X^TX)^{-1}X^Ty$$

They require the inversion of $X^TX$. This can lead to problems if the system of euqations is poorly conditioned. A solution is to **add a small element to the diagonal** $\delta^2I_d$, but there is interesting findings, after adding the small element, the model result $\bar Y$ for the form $\hat\theta$ (adding a small element) is better than the original form $\hat\theta$ (without adding the small element).
$$\hat\theta = (X^TX + \delta^2I_d)^{-1}X^Ty$$

The interesting findings led the math scientists to do the research, they find the definition of $\theta$ is to get the  $argmin J(\theta) = (y-X\theta)^T(y-X\theta)$, 

This is the **ridge regression** estimate. It is the solution to the following **regularized quadratic cost function**
$$J(\theta) = (y-X\theta)^T(y-X\theta)+\delta^2\theta^T \theta$$

Because we are looking for the argmin $J(\theta)$, so we do partial derivative to $\theta$, then we have following

**Derivation**
$$\begin{aligned} \frac{\partial}{\partial\theta} J(\theta) &= \frac{\partial}{\partial\theta}  \{ (y-X\theta)^T(y-X\theta) + \delta^2\theta^T \theta\}  \\
& = \frac{\partial}{\partial\theta}  \{ (y^Ty-2y^TX\theta) + \theta^TX^TX\theta + \theta^T (\delta^2I)\theta\} \\
&=-2X^Ty + 2X^TX\theta + 2\delta^2I\theta \\
&=-2X^Ty+2(X^TX+\delta^2I)\theta
\end{aligned}$$
Equating to Zero, yields (definition of derivation, the extremum value could be found at partial derviative euqal to zero, and since it is a convex fucntion, so it has global optimal solution)
$$\hat\theta_{ridge} = (X^TX + \delta^2I)^{-1}X^Ty$$

**Attention**: $I$ represents identity matrix.

**Conclusion** on ridge regression formula, if the $J(\theta)$ shoul$d be argmin, then the value of $\delta^2\theta^T \theta$ could not be large, do example for $X$ with 5 feature, then we have $[X_1, X_2, .... X_5]$, then the $\theta$ has 5 dimension vector $[\theta_1, \theta_2, ... \theta_5]$, the $\delta^2\theta^T$ to be $\delta$*$[\theta_1^2+\theta_2^2+ ... +\theta_5^2]$, the $\delta$ is an constant.

From its Geometry explaination (convex optimization)

**Ridge regression as constrained optimization**
![ridge regression](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/ridge%20regression2.PNG)

When the $(y-X\theta)^T(y-X\theta)$ is small, the $\delta^2\theta^T \theta$ will large, else it is opposite, so the formula try to find a balance value. $\delta^2$ is the super parameter, so to get $J(\theta)$ can transform the formula to argmin, let  constrain condition has $\theta^T \theta \le$ constant $t(\delta)$ .  If $\theta$ in 2D ($\theta = (\theta_1, \theta_2))$, then $\theta^T \theta = \theta_1^2 + \theta_2^2$. Suppose $\theta_1^2 + \theta_2^2 \le$ constant, plot it on 1D is concentric circles. $\delta$ value decides the best solutions. 

### Ridge, feature selection, shrinkage and weight decay
Large values of $\theta$ are penalised. We are $shrinking \space \theta$ towards zero. This can be used to carry out $feature \space weighting.$ An input $x_{i,d}$ weighted by a small $\theta_d$ will have less influence on the output $y_i.$ This penalization with a regularizer is also known as weight decay in the neural networks literature.

Note that shrinking the bias term $\theta_1$ is undersirable. To keep the notation simple, we will assume that the mean of $y$ has been subtracted from $y.$ This mean is indeed our estimate $\hat\theta_1.$

### Selecting features for prediction
![selecting features](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/selecting%20features%20for%20prediction.PNG)

Ridge has a big disadvantage, suppose $x$ has 500 dimensions $x_{500}$, then the shrinkage $\theta$ could not be zero, because just few of $\theta_i$ could be shrinked, eg $\theta_1, \theta_2,...$, becuase the ridge did not do feature selection. Since some of features could be selected in $x$ by randomly, so the necessary **first step** will be **combination of good features** (remove not necessary features), this is what **LASSO  did**.

#### Compare Ridge and LASSO
![enter image description here](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/ridge%20and%20lasso%20compare.PNG)

**In Ridge**, when we **increase** $\delta^2$, the $\theta_i$ has a trend towards zero, but **it will not reach to zero, only near to zero**. **In LASSO**, **increase** the $\delta^2$, some $\theta_i$  are immediately changed to zero, such  $\theta_i$ could be removed from the features, do example, there are eight $\theta_i$, two of $\theta_i$ are jumped to zero when increase the $\delta$, so just keep the rest of six $\theta_i$, which means there are eight feature dimentions are decreased to six feature dimentions.  

#### How LASSO work?
![enter image description here](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/How%20LASSO%20work.PNG)

Suppose we want get $J(\theta) = (y-X\theta)^T(y-X\theta)$, in ridge regression, we have $J(\theta) = (y-X\theta)^T(y-X\theta) + (\theta_1^2+\theta_2^2...\theta_d^2)$, for the  $\sum_{j=1}^d \theta_j^2$ **(Another name is L2)** could be changed to  $\sum_{j=1}^d |\theta_j|$ **(Another name is L1)**, the concentric circles are changed to concentric rhombus. The intersection between rhombus and oval only possible at axis, the proof is not hard, you can find the detailed demonstration steps in any statistics book. Changing the $\delta^2$ (decrease or increase), the rhombus will be zoomed in or out. When the rhombus intersected with oval on $\theta_1$ axie, the $\theta_2$ will **keep in zero always**. So using LASSO,  features which relate to $\theta$ can be changed to 0, like $x_i$->$\theta_i$->0.  

The optimized solutions in Riger regression will be found on the tangent of circle and ovral, in LASSO will be found on axis of rhombus and ovral when they intersected.  All $\theta$ together to be zero in riger regression,  all kinds of $\theta$ to be zero with different speed in LASSO, this is why using LASSO could do features selection, the Riger regression could not.
### Going nonlinear via basis functions
![going to nonlinear](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/going%20to%20nonlinear.PNG)

It manually add the input data $x$ with extra $x^2$, then the linear line will change to non-linear, then the **hyperplane (linearly)** will be changed to **curved surface (non-linearly)**.

![enter image description here](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/linear%20change%20to%20nonlinear.PNG)

**Example:** Ridge regression with a polynomial of degree 14
![enter image description here](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/ridge%20with%20polynomial%20of%20degree%2014.PNG)

Suppose we change the input $x$ to be a polynomial of degree 14, as $x^{14}$, then we need add the extra L2 norm ($\delta^2\theta^T\theta$). Similar problem, if the **$\delta^2$ very small**, then the L2 norm nearly to zero. Then doing the polynomial of degree 14 without penarity (L2 = 0), it will be **overfitting**. if the **$\delta^2$ very large**, the $\theta$ will close to zero. it makes **underfitting**. The medium $\delta$ will have the best performance in ridge regression, but problems again, **how chose the best $\delta$?**

#### Kernel regression and RBFs
We can use kernels or radial basis functioins (RBFs) as features:
$\phi(x) = [k(x,\mu_1,\lambda),...,k(x,\mu_d,\lambda)], e.g. \space k(x,\mu_i,\lambda) = e^{(-\frac{1}{\lambda}||x-\mu_i||^2)}$

![enter image description here](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/kernel%20method.PNG)

In previous example, the linear change to non-linear using the naive method, like simply with a polynomial of multiple degrees. Now the kernel method is adding bell shape (look like normal distribution) on each input $x$, it is kernal function$\space k(x,\mu_i,\lambda) = e^{(-\frac{1}{\lambda}||x-\mu_i||^2)}$, it is controlled by $\mu_i$ and $\lambda$. $\mu_i$ is the mean symmetry point, the $\lambda$ decides the width size of shape bell, so the $y$ is the linear regression of all kinds of kenels (like $\hat y = \theta_1 k_1+\theta_2 k_2+...\theta_d k_d$).

Regarding to kernel function, there are other questions:
-  How to decide the number of kernels (eg. 8 kernels or 10 kenels)
- When know the total number of kernels, then where to put them? (the location on axis)

**Trick base on experience**, at the beginning of axis $x$, a kernel will be added, then we do k-means clusting, which can help decrease the complexity of $\hat y$ linear functions. Do example, suppose there are input $x$ which has [10,000*50] dimensions,  transform to kernel, there are 10,000 kernels ($k_1, k_2, ....k_{10,000}$), then the computation of $\hat y = \theta_1 k_1+\theta_2 k_2+...\theta_{10,000} k_{10,000}$ will take long time. So **use** **k-means** method, **define the number of $k$** (kernel), there are number of $k$ clusters to gather the input $x$, that will decrease the number of kernels and easy to do computation of $\hat y$.   Above theoretical mechanism with two name, RBF (radius brief function) and kernel. It is the same methodology of kernel in Gaussian regression. The kernel is used to calculate the similarity, in SVM theory, if the **kernel is following positive semi-definite matrix rules**, then **it will be good** one. Example, ${x\cdot y}$ is 0, then means $x$ and $y$ are totally difference, if the value it very large, means they have a lot of similarites. 

The kernel used in SVM, is transform the input value into hyper space, after dot product ($\cdot$) return back, it only concerns the result in after dot product, not the transform itself. 

We can choose the locations $\mu$ of the **basis functions** to be the inputs. That is, $\mu_i = x_i$ These basis functions are known as **kernels**.  The choice of width $\lambda$ is tricky, as illustrated below.

![kernel parameter](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/kernel%20parameter.PNG)

Too small $\lambda$ will cause of overfitting, to large $\lambda$ will cause of underfitting. Right $\lambda$ will output with best performance. 

Base on above detailed explanation, they are all direct to same question, how to find proper value of $\delta^2$, $\lambda$ , it is the big problem, and this is **engineering problems**.

### The big question is how do we choose the regularization coefficient, the width of the kernels or the polynomial order?
How to choose the proper hyperparameter?

- Width of kernels $\lambda$
- Polynomial order, eg. polynomial degree in 4 ($x^4$) or 14 ($x^{14}$). 
 There is no special experience for polynomial order, according to occam's razor principle, the simple is the best, if using ploynomial degree in 3 ($x^3$) could output better peformance, then there si no reason to use higher ploynomial degree more than 3 ($x^3$). 
  
 Using cross-validation method.
 #### Method 1
  Suppose we have only one value for parameter setting
  1) Give value of $\delta^2$, then choose an learning algorithm (eg. LASSO) 
  2) Using training dataset (70%) to training the model
  3) If the model from training dataset is good, then using test dataset to test the model
  4) If the mode from test dataset (30%) is also good,
  5) Then using training + test dataset to train a final model
  6) Base on final model to do backtesting

#### Method 2
Suppose we have multiple possible values for parameter setting, eg $\lambda^2 = 1 \space or \space 0.5 \space or \space 0.01$)
 1) Give multiple value of $\delta^2$, then choose an learning algorithm (eg. LASSO) 
  2) Using training dataset (70%) to training multiple models which with different input value of $\lambda^2$, then we get different model (eg. Model 1, Model 2, Model....)
  3) Then using validation dataset (15%) to test for different models, choose the best performance model.
  4) Then using training dataset (70%) + validation dataset (15%) with value of $\delta^2$ which comes from step 3) the best performance model's parameter to training another model.
  5) Then using test dataset (15%) to test the model (which created in step 4) performance. 
  6) If the performance model result in step 5 is good, then using 100% dataset (70% training dataset + 15 validation dataset + 15% test dataset) to train a final model.
  7) Base on final model to do backtesting

For the possible of hyperparameter's setting value, it could be a lot, eg. for the parameters, $c= {0.5, 1, 1.5...}, \gamma ={0.01, 0.5, 0.75...}$ which setting for SVM model, you can manually try all kinds of values setting. In Sklearn, it provides a Grid Search to help you find the best parameters combination. Base on heat map, you can find the best parameters combination. 

![heat map](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%204/IMG/heat%20map.PNG)

If the superparameters are more than two, then the heat map could not plot all of them in one map, we have to plot each two of them, then this bring another problem, how to parallel processing to speed up calculation and optimize the algorithm to reduce the complexity. 

## Now, big question
- How to define input $x$?
The list of features in $x$ are critical for quant analysis, there are thousand methods to tell what should be defined as features (indicators), or you can define base on your experience. If you are not familiar with financial indicator, here is the good website to help you study: [list of financial technical indicators](https://school.stockcharts.com/doku.php?id=technical_indicators).

# Homework
- Assignmet 1: Write your function, using the technical indicators $x$ to output $y$.

