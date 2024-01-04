from src.process import *
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier, Ridge, LinearRegression, Lasso, ElasticNet,  LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier


Methods = [
            LogisticRegression(penalty='l1', C=0.01, solver='liblinear', max_iter=200, n_jobs=-1),
            #RandomForestClassifier(n_jobs=4, n_estimators=250, class_weight='balanced_subsample', max_features=4, max_depth=10, min_samples_leaf=8),
            #AdaBoostClassifier(),
            #BaggingClassifier(n_jobs=4, n_estimators=250)
            ]

regresive=[0,1,2,3,4,5,6,7,8,9,10,11,12]
values_cbs=[1,1,1,1,0,0,0,1,1,1,0,0,0]
regresive=[regresive[x] for x in range(len(regresive)) if values_cbs[x]==1]
cuartiles = [90,90,90,90]

objs = system_preparation(Methods_to_use=Methods, regresive_method=regresive)
exit()
objs = bining_and_targets(objs)
objs = feature_normalization(objs)
objs = linkage_analyses(objs,cuartiles)

# print('*'*30)
# print("***** In MAIN")
# for count,test in objs[0].fairness_metamorphic.items():
#     print("***** ",count)
#     print("***** ",test)

### metamorphic evaluation and data augmentation
objs=metamorphic_evaluation(objs,mbd=True, mbc=False, mbe=False)
# print('*'*30)
# print("***** In MAIN")
# print("***** ",objs[0].fairness_metamorphic_results)
exit()
objs=data_augmentation(objs)
objs=run_experiments(objs)
objs=metamorphic_evaluation(objs,mbd=False, mbc=False, mbe=True)

#Print Final results
objs=show_and_tell(objs)



