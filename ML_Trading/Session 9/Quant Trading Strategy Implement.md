# Machine Learning and Quantitative Trading - Session 9
Top conferences:
[AAAI-AI](https://www.aaai.org/home.html), [ICML-ML](https://icml.cc/), [IJCA](https://dl.acm.org/journal/ijca), [NIPS-Neuro](https://nips.cc/), [KDD-Data Mining](https://www.kdd.org/)

## Eigenvalues & Feature Vector
It is the good method to describle an object in a matrix by eigenvalues and feature vector.  eg, in the formula, $(A-\lambda I)\bar X=0$, $\lambda$ is the eigenvalues, $\bar X$ is the feature vector

For $A=A^T$, the symmetric matrix with following theorem:
- $\lambda \in$ R
- $\lambda_i$ and $\bar X_i$ are perpendicular to each other

For non-symmetric matrix A
If there is eigenvector S exist (means it has $S^{-1}$), then we have following formula:
$A= S\sigma S^{-1}$, S is the vector $[\bar X_1, \bar X_2, ...]$, $\sigma$ is the diagonal matrix

It could be inference according to matrix define: $AX = \lambda X$
So $AS=A[\bar X_1, \bar X_2..\bar X_n]$
     $AS=\lambda [\bar X_1, \bar X_2..\bar X_n]$
     $AS=[\lambda_1\bar X_1, \lambda_2\bar X_2..\lambda_n\bar X_n]$
     $AS=[\bar X_1, \bar X_2..\bar X_n]$ * 
|$\lambda_1$  |...|
|....|$\lambda_2$|
|....|......| ......|
| ... |...  |.....|$\lambda_n$|
So, the $A=S\lambda S^{-1}$

Recommend online course: [Linear Algebra - MIT](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)

SVD is the method to help decompose any matrix A to be $A=U\sum V^T$ , U is the orthogonal matrix, $\sum$ is the diagonal matrix, and $V^T$ is orthogonal matrix again. In python, input the A, the SVD method will output U, $\sum$ and V. Orthogonal matrix U is inveriable $U^{-1}$, inverse matrix $U^{-1}$ is equal to its transform matrix $U^T$.

When we have the return of stock price at time t, then we have return vector, $\bar r = [r_1, r_2,...r_N]$,  then we can get the covariance matrix $\sigma$. Then we have different ratio of the stock, so we have ratio vector at time ta, $\bar w=[w_1, w_2,...t_n]$, then we have the formula for risk, $RISK = \bar w \sigma \bar w^T$, according to the risk formula, you can find the eigenvalue and eigenvector, the **min $\bar w$** =$\bar x$ is the target for every investor (**min RISK** = min $\lambda$ value) , so there is relatives between $\lambda$ and $\bar x$.


Recommend online course, [Convex Optimization - Standford](https://www.bilibili.com/video/av837646777/)

The procedure for RISK calculation:
1. Get the covariance vector of $\sigma$, then find the related eigenvector $\bar x$
2. Using the random matrix theroem, find the mean of distribution value (eg, 0.5) of the covairance vector, set the variance $\sigma$ to zero in range [0.2-0.8]
3. Using the new covariance vector value get $\sigma' = Q \Sigma Q^T$
4. Then we have RISK = $w^T\sigma' w$

Using the fiter trick to remove the random value in covariance, then the $\bar w$ with less random value, the result will be more accurate.

Recommend paper for quant trading
1. Classification-based Financial Markets Prediction using Deep Neural Networks - 2017
2. Deep Learning for Multivariate Financial Time Series - 2015
3. Quantifying the effects of online bullshiness on international financial markets - 2015
4. Predictive analysis of financial time series - 2014

Recommend desktop configuration:
i7+64g ram+1080 GPU and screen (U2415 DELL 23 16:10) x2 or 4T GPU + 128g ram
