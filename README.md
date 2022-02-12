# BoostingWithGoss
This repo holds an implementation of Gradient Boosting Decision Tree which aims to reproduce the Gradient one-side sampling technique presented in LightGBM

Light gradient boosting machine (LightGBM) is an integrated algorithm for building gradient boosting decision tree(GBDT), which has the characteristics of faster trainingspeed, lower memory consumption, better accuracy, andsupport for parallel processing of massive data. Different from tradi-tional algorithms for generating GBDTs, such as XGBoost, pGBRT, scikit-learn, etc., LightGBM mainly optimizes the following aspects:

## Gradient-based One-Side Sampling (GOSS)

GOSS mainly realizes data sampling. Since large gradientsamples have a greater impact on information gain, GOSS discards samples which are not helpful in calculating infor-mation gain. When data sampling is performed, only large gradient instances are retained, and small gradient instances are randomly sampled while introducing constant multipliers (1−a)/b, leading to make the algorithm pay more attention to the instances of insufﬁcient training and reduce the impact onthe distribution of the original dataset.

![Goss flow](goss-flow.png "Goss flow")
