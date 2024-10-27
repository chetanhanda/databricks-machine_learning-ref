# Databricks notebook source
import pyspark.pandas as pd
import numpy as np

# COMMAND ----------

files = dbutils.fs.ls("/mnt/expt-sklearn")
display(files)

# COMMAND ----------


pys_Df = spark.read\
    .option("header",True)\
    .option("inferenceSchema",True)\
    .csv("dbfs:/mnt/expt-sklearn/heart-disease.csv") 

display(pys_Df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### to pyspark panda

# COMMAND ----------

pys_Pd_Df = pys_Df.toPandas()
display(pys_Pd_Df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### get the features and the target variables

# COMMAND ----------

# X = features matrix
X = pys_Pd_Df.drop('target', axis=1)

# y = labels
y = pys_Pd_Df['target']

# COMMAND ----------

# MAGIC %md
# MAGIC #### select the model

# COMMAND ----------

# using a classification ml model and hyper parameters
from sklearn.ensemble import RandomForestClassifier

# default hyperparams
clf = RandomForestClassifier()

# see what the default hyper params are 
clf.get_params()

# COMMAND ----------

# MAGIC %md
# MAGIC #### split the data

# COMMAND ----------

# get the  data split up 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### fit the model to the data 

# COMMAND ----------

clf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### try making a prediction

# COMMAND ----------

y_preds = clf.predict(X_test)
y_preds

# COMMAND ----------

# MAGIC %md
# MAGIC #### evaluate the model with training data : should be 1 

# COMMAND ----------

# score will be perfect because its the training data
clf.score(X_train,y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### evaluate the model with testing data 

# COMMAND ----------

clf.score(X_test,y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### print the performance report

# COMMAND ----------

from sklearn.metrics import classification_report,confusion_matrix  ,accuracy_score

print(classification_report(y_test,y_preds))

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

confusion_matrix = confusion_matrix(y_test,y_preds)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - True Negative (Top-Left Quadrant)
# MAGIC - False Positive (Top-Right Quadrant)
# MAGIC - False Negative (Bottom-Left Quadrant)
# MAGIC - True Positive (Bottom-Right Quadrant)

# COMMAND ----------

# MAGIC %md
# MAGIC #### accuracy score

# COMMAND ----------

accuracy_score(y_test,y_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC ####  hyper parameter tuning for better accuracy with hyperopt

# COMMAND ----------

from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

space = {
        'criterion': hp.choice('criterion', ['entropy', 'gini']),
        # 'max_depth': hp.quniform('max_depth', 10, 1200, 10),
        'max_features': hp.choice('max_features', ['sqrt','log2', None]),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
        'n_estimators' : hp.choice('n_estimators', [10, 20,30,40,50,60,70,80,90,100])
    }

def objective(space):
    model = RandomForestClassifier(
                                 criterion = space['criterion'],
                                #  max_depth = space['max_depth'],
                                 max_features = space['max_features'],
                                 min_samples_leaf = space['min_samples_leaf'],
                                 min_samples_split = space['min_samples_split'],
                                 n_estimators = space['n_estimators'] 
                                 )

    accuracy = cross_val_score(model, X_train, y_train, cv = 5).mean()
    # We aim to maximize accuracy, therefore we return it as a negative value
    return {'loss': -accuracy, 'status': STATUS_OK }



trials = Trials()

best = fmin(fn= objective,
            space= space,
            algo= tpe.suggest,
            max_evals = 100,
            trials= trials)
best  

# COMMAND ----------

# MAGIC %md
# MAGIC #### look at the results and choose the best value of the hyperparameter which gave the best accuracy 
# MAGIC ##### retrain the model with the specific hyper parameter

# COMMAND ----------

clf = RandomForestClassifier(n_estimators=3,max_features=1,min_samples_leaf=0.2429998889267651,min_samples_split=0.025696850647164204).fit(X_train,y_train)


# COMMAND ----------

# MAGIC %md
# MAGIC #### check if the accuracy of new model is as expected using the test data

# COMMAND ----------

print(f"Model accuracy on test set : {clf.score(X_test,y_test)*100:2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### save the model

# COMMAND ----------

import pickle

pickle.dump(clf,open("random_forest_model_1.pkl","wb"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### load the model & see how it performs

# COMMAND ----------

loaded_model= pickle.load(open("random_forest_model_1.pkl","rb"))
loaded_model.score(X_test,y_test)

# COMMAND ----------


