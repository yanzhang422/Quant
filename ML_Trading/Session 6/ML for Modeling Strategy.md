# Machine Learning and Quantitative Trading - Session 6
**Outline**
 - Feature Selection
 - Genetic Algorithm
 - Deep understanding BP
 - RNN

## Feature Selection
- **Subset Selection**. We identify a subset of the p predictors that we believe to be related to the response. We then fit a model using least squares on the reduced set of variables. Here is the random selection the features $\bar X [x_1, ...., x_p]$ by yourself, it could be based on your experience, preference,etc.
- **Shrinkage**. We fit a model involving all p predictors, but the estimated coefficients are shrunken towards zero relative to the least squares estimates. This shrinkage (also known as regularization) has the effect of reducing variance and can also perform variable selection. We choose all the dimensions, but some dimension/feature which is important less, then we shrink it to be zero. This is what LASSO do.
- **Dimension Reduction**. We project the p predictors into a M-dimensional subspace, where M < p. This is achieved by computing M different linear combinations, or projections, of the variables. Then these M projections are used as predictors to fit a linear regression model by least squares. Using a function to convert the input $x_i$ as a vector ($x_{300x1}$) to be $z_{20}$, the 20 is the combination of linear or non-linear of the 300 dimensions. So the output result (reduced dimension) with difference meaning for input dimension. eg. PCA, SVD, Auto Encoder is one of the method for dimension reduction.

### Subset Selection
If we have 300 features, but we don't know which features are better without support by expert or experience. So only can use greedy search method with following steps:

Best subset and stepwise model selection procedures
1. Let $M_0$ denote the null model, which contains no predictors. This model simple predicts the sample mean for each observation.
2. For k = 1,2,...p: Here is using permutations, like  $\lgroup _2^{300} \rgroup$ $\lgroup _3^{300} \rgroup$....
    (a) Fit all $\lgroup _k^p \rgroup$ models that contain exactly k predictors.
    (b) Pick the best among these $\lgroup _k^p \rgroup$ models, and call it $M_k$. Here best is defined as having the smallest RSS, or equivalently largest $R^2$
3. Select a single best model from among $M_0, ...., M_p$ using corss-validated prediction error, $C_p$ (AIC), BIC, or adjusted $R^2$.

**This method could be used for less dimensions, if you have many dimension, the greedy search will be stucked in computer running.**

#### Forward Stepwise Selection
It is an O(n) computer complexity.

- Forward stepwise selection begins with a model containing no predictors, ad then adds predictors to the model, one-at-a-time, until all of the predictors are in the model.
- In particular, at each step the variable that gives the greatest additional improvement to the fit is added to the model.

Forward Stepwise Selection in detail: 
1. Let $M_0$ denote the null model, which contains no predictors.
2. For k=0,..., p-1:
    2.1 Consider all p-k models that augment the predictors in $M_k$ with one additional predictor.
    2.2 Choose the best among these p-k models, and call it $M_{k+1}$. Here best is defined as having **smallest RSS or highest $R^2$**.
3. Select a single best model from among $M_0,...,M_p$ using cross-validated prediction error, $C_p$ (AIC), BIC or adjusted $R^2$.

## Genetic Algorithm (GA)
GA is super important, it is not just used in feature selection, also be used for future stock selection. 
How to select the good features?
There are several ways:
1. Probability Selection
   The individual feature's fitness/sum of all features' fitness as $P_1 = \frac{f_1}{\sum|f_i|}$
2. Rank Selection
3. Base on fitness and diversityto do selection
    The diversity could be calculated on variance or entropy. Just like the DT which seperate the group of data, try min the diversity inner the groups, try max the diversity between each group which is seperated by entropy.
    
The GA could be used for the questions of cryptography, eight queens problem and stock selection. 

Disadvantage: 
1. When you have less features, then it hard to find the global optimal value. Better use it for with large features, parallel running and thousand iterations.
2. It is hard to be predicted, just like NN.

Recommend to read PPO algorithm, it relevants to GA. [PPO Algorithm](https://zhuanlan.zhihu.com/p/182578237)
And the 2019 paper list  in quick reviewing, [2019 Paper Digest](https://www.paperdigest.org/2019/11/neurips-2019-highlights/)

### ANN
CNN is used for image recognition, the BP will be the key method for NN and for quant trading. 
The derivation of weight of neuros to get the result of weight $w$.

![Derivation of weight of neuros](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%206/IMG/Derivation%20of%20weight.PNG)

Then we do the reverse of the chain rule for above derivation of the $w_1$ and $w_2$. And if we have multiple input x ($x_1, x_2$), then the output will be mutlple of z ($z_1, z_2$).

![Reverse of the chain rule](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%206/IMG/reverse%20of%20the%20chain%20rule.PNG)

When we have multiple input x, and 1:n connection of the neuros, then we have exponential complexity of the chain rule connection. Means when we increase the layer in linear, the possible path is increased in exponential. 

![BP help decrease the computer complexity](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%206/IMG/BP%20help%20decrease%20the%20complexity.PNG)

**The essential of BP method, it solves this problem.** It just derivates once for the same factorization, then reuse the same result for the other derivation of the weight. This is the chain rule + dynamic programming method. 

![Essential of BP](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%206/IMG/Essential%20of%20BP.PNG)

Recursive NN is used in NLP, Recurrent NN is used for time series of data.

An interesting video regarding to Genetic evolving animation by Karl Sims - Evolved Virtual Creatures, Evolvution Simulation. [Evolved Virtual Creatures](https://www.youtube.com/watch?v=JBgG_VSP7f8)

Recommend book lists by Ernest P. Chan:
[Algorithmic Trading: Winning Strategies and Their Rationale (Wiley Trading)](https://www.amazon.com/gp/product/B00CY5HC0U/ref=dbs_a_def_rwt_hsch_vapi_tkin_p1_i0)
[Quantitative Trading: How to Build Your Own Algorithmic Trading Business (Wiley Trading Book 381)](https://www.amazon.com/gp/product/B001FA0GGC/ref=dbs_a_def_rwt_hsch_vapi_tkin_p1_i1)
[Machine Trading: Deploying Computer Algorithms to Conquer the Markets (Wiley Trading)](https://www.amazon.com/gp/product/B01N7NKVG0/ref=dbs_a_def_rwt_hsch_vapi_tkin_p1_i2)



