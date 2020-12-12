# Machine Learning and Quantitative Trading - Session 7

## Data Analysis
Analysis the data whether 
1) Class balanced (Most finance data is unbalanced)
   - Stastistics
   - Visualization
2) Select the classifier/Regressor to do further analysis for the result in Step 1
   - Try more
   - Think more
   - Parameter Space understanding
3) Select Features
    - Long term feature (General data)
    - Short term feature (General data)
    - Customized feature (eg. NLP)

## Ensemble Learning
This is the key skills for every quant trader to be mastered. Most of the data could be extract online, source code of models are free, but ensembling is very important skill to improve the accurate performance by several models combining.

1) Feature Extractions
2) Modelings $[M_1, M_2...M_n]$
3) Ensemble those modelings
4) Strategy (when you have a model with higher confidence, then how to start investing with this model, it bases on Kelly ratio)

 - Stacking/Blending/Voting
   Easy, but important. Eg, if you have 10 different classifiers, they are SVM, RF, LR...etc. Then we have new data input x, different classifier with different output Y (+1, -1), then we can use the **hard voting method**, sum of all the classifier's prediction result.  
![hard voting](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%207/IMG/hard%20voting.PNG)
 But hard voting did not consider different classifier with different accuracy, the higher accuracy classifier should have higher confident performance, so it should have higher weight in the voting. This is the blending key method:
 $H=\alpha_1h_1+\alpha_2h_2+.....+\alpha_nh_n$, then we have another question, **How to define the weight value $\alpha$ for each classifier?** There are several definition methods:
      1) Which classifier with higher accuracy, then with higher weight, eg, ACC/$\sum_{ACC}$
      2) Machine Learning method, according to the prediction value of each classifier, eg ($\hat y_1, \hat y_2,...\hat y_n$) with actual value y, then we learned the weight value for each classifier from linear function $f (\alpha_1h_1+\alpha_2h_2+.....+\alpha_nh_n)$
![predict weight value by ML](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%207/IMG/predict%20weight%20value%20by%20ML.PNG)
 - **Adaboost:** Adaptive Boosting (it will not overfitting)
    We have many weak classifiers, the model accuracy just above 50%, eg 55%, 58%.. the adaboost teach you how to ensembling such weak classifier to a strong classifier, and it has been theoretically proved.
    1. Add weight in each sample, because in loss function, it squares the prediction and actual value, but without weight for individual sample. So each sample with the same contribution on the loss function, but in fact, it is not the truth. Some sample could be very important and impact on loss function a lot, so need weight value to reflect it.
    $min Loss (\theta) = \frac{1}{N} (\alpha_i \hat Y_{(\theta)} - Y)^2$
    Here is the procedure by Adaboost:
    - Training the classifier from original test data ($h_1$)
    - Add weight strength $\alpha$ on $h_1$ for part of the data with wrong classification, the weight strengthed "new data" as training data to generate another classifier ($h_2$)
    - Add weight strength $\alpha$ for the difference partial between $h_1$ and $h_2$ with wrong classification, the weight strengthed "new data" as training data to generate another classifier ($h_3$)
    - Finally, calculate the $H=\alpha_1 h_1+\alpha_2 h_2+\alpha_3 h_3$
   2. Sub model, could be seperate to sub-sub model according to the same method on H, eg, $h_1$ could be combined by $h_{11}$,  $h_{12}$, $h_{13}$, $h_2$ could be combined by $h_{21}$,  $h_{22}$, $h_{23}$, etc.
   3. Analysis the tree with just two branch, node+leaves.
   4. Add weight $W_i$ on different kind of classifiers, we will have  $Y=X_iW_i$

Following is the Adaboost calculation steps:
1. Initialize $W_i$,  $W_i = \frac{1}{N}$, 
2. Then we pick the top t best $h_t$ from the random generated classifiers $h$ according to minize $\eta_t$, the weight $\alpha_t$ for each classifier is calculated, $H=(\sum \alpha_ih_i)$
3. Then calculates the $W_i^{t+1} = \frac{W_i^t}{2}*e^{-\alpha^t h^t(x)^*Y_i}$, loop in step 2 and 3.

**Adaboost will not overfitting**, Sklearn with adaboost library, if you can define its base estimate. Recommend read in example of Sklearn->ensemble->Adaboost->User guide, [Adaboost Example](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
   
 When you create your confident model:
 1. Extra data
 2. Data features
 3. Modeling
 4. Ensemble

Then the next step is how to invest your money in the model:
 1. Risk Control
     Furture prediction, this is data science area in math problem.
 2. Strategy
     Base on your prediction, how to max your profit, this is pure finance problem.
