import os
import pickle
import sys
import numpy as np
import yaml
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import json
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

params = yaml.safe_load(open("params.yaml"))["SVMtrainEvaluate"]

split = params["split"]
seed = params["seed"]
features= params["max_features"]
input = sys.argv[1]

data_input = os.path.join(input,"features.csv")

data_input = pd.read_csv(data_input)

df = pd.DataFrame(data_input)


X = df['text']

y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = split, random_state =seed)

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=features)
vectoriser.fit(X_train)
X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)


def model_Evaluate(model):
 
 # Print the evaluation metrics for the dataset.
 print(classification_report(y_test, pred_y))
 # Compute and plot the Confusion matrix
 cf_matrix = confusion_matrix(y_test, pred_y)
 categories = ['Negative','Positive']
 group_names = ['True Neg','False Pos', 'False Neg','True Pos']
 group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
 labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
 labels = np.asarray(labels).reshape(2,2)
 sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
 xticklabels = categories, yticklabels = categories)
 plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
 plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
 plt.title ("SVM Confusion-Matrix", fontdict = {'size':18}, pad = 20)


SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
pred_y = SVCmodel.predict(X_test)
model_Evaluate(SVCmodel)

output = sys.argv[2]

with open(output, "wb") as fd:
    pickle.dump(SVCmodel, fd)

## Plot roc curve
fpr, tpr, thresholds = roc_curve(y_test, pred_y)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC-CURVE')
plt.legend(loc="lower right")
plt.show()


acc = cross_val_score(SVCmodel, X_test, y_test, cv=5, scoring='accuracy')

print("SVM Sentiment Analysis Accuracy:\t")
accuracy = acc.mean()
print(accuracy)

average_precision_score = metrics.average_precision_score(y_test, pred_y)
roc_auc_score = metrics.roc_auc_score(y_test, pred_y)

# Calculating R-squared error:

r_squared = SVCmodel.score(X_test, y_test)

# Calculating Root mean squared error:

rmse = math.sqrt(mean_squared_error(y_test, pred_y))


metrics_output = os.path.join("data", "Metrics", "SVM_metrics.json")

os.makedirs(os.path.join("data", "Metrics"), exist_ok=True)

with open(metrics_output, "w") as outfile:
    json.dump(dict(accuracy = accuracy, r_square_error = r_squared, root_mean_squared_error = rmse, average_precision_score = average_precision_score,roc_auc_score = roc_auc_score), outfile)

