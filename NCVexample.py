import numpy as np
from sklearn.model_selection import StratifiedKFold
import itertools

from sklearn import model_selection,metrics 
from sklearn.metrics import accuracy_score
import xgboost
import time
import sklearn.datasets

X,Y = sklearn.datasets.make_classification(n_samples=1000)

# data is split in a stratified fashion into train and test sets
seed = 7
test_size = 0.30
X_train_final, X_test_final, y_train_final, y_test_final = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)



outerK = 4
innerK = 2
early_stopping_rounds = 100
eval_metric = 'auc'

skf = StratifiedKFold(n_splits=outerK)
skf.get_n_splits(X_train_final, y_train_final)

parameters = {'nthread' : [-1],
              'objective':['binary:logistic'],
              'learning_rate': [0.1], #so called `eta` value
              'max_depth': [6,10,14],
              'min_child_weight': [1,3,7,10,13],
              'silent': [1],
              'n_estimators': [1000000], #number of trees, change it to 1000 for better results
              'seed': [1337]}



def nested_CV_GS_XGB(parameters):
    
    def my_product(dicts):
        product = [x for x in apply(itertools.product, dicts.values())]
        return [dict(zip(dicts.keys(), p)) for p in product]
        
    bestModels = []
    bestModelScores = []
    bestModelsOuterparams = []
    topModels = []
    outerCounter = 1
    for train_index, test_index in skf.split(X_train_final, y_train_final):
        X_train, X_test = X_train_final[train_index], X_train_final[test_index]
        y_train, y_test = y_train_final[train_index], y_train_final[test_index]
        
        skfInner = StratifiedKFold(n_splits=innerK)
        skf.get_n_splits(X_train, y_train)

        tempModels = []
        scores = []
        innerCounter = 1
        for params in my_product(parameters):
            model = xgboost.XGBClassifier(**params)
            for train_index_inner, test_index_inner in skf.split(X_train, y_train):
                print 'inside inner loop'
                X_train_inner, X_test_inner = X_train[train_index_inner], X_train[test_index_inner]
                y_train_inner, y_test_inner = y_train[train_index_inner], y_train[test_index_inner]
                model.fit(X_train_inner, y_train_inner,verbose = False, early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, eval_set=[(X_test_inner, y_test_inner)])
                scores.append(model.best_score)   
            avgScore = float(sum(scores))/len(scores)
            tempModels.append([model.get_params(), avgScore])
            
            print 'Finished Inner Fold For Some Inner Model', innerCounter
            innerCounter+=1
        tempModels.sort(key=lambda x: int(x[1]))
        bestMod = tempModels[-1][0]
        bestModels.append(bestMod)
        

        
        outerModel = xgboost.XGBClassifier(**bestMod)
        outerModel.fit(X_train, y_train,verbose = False, early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric,
              eval_set=[(X_test, y_test)])

        bestModelsOuterparams.append([[bestMod], [outerModel.best_score]])
        bestModelScores.append(outerModel.best_score)
        topModels.append(outerModel)
        print 'Finished Outer Fold', outerCounter, 'Score:', outerModel.best_score
        outerCounter+=1
    avgBestModelScores = float(sum(bestModelScores))/len(bestModelScores)
    
    return avgBestModelScores,topModels,  bestModelsOuterparams


t0 = time.time()
roc_auc, topModels, bestModels = nested_CV_GS_XGB(parameters)
t1 = time.time()
print 'function took', float(t1-t0)/60, 'mins' 
print 'Estimated ROC_AUC:', roc_auc

           
for i in range(outerK):
    print i, 'model max_depth:', bestModels[i][0][0]['max_depth']
    print i, 'model min_child_weight:', bestModels[i][0][0]['min_child_weight']
