#!/usr/bin/python

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import pickle
import sklearn
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.preprocessing import Imputer

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'long_term_incentive', 'other',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Convert dictionary to a dataframe
enrondf = pd.DataFrame.from_dict(data_dict, orient = 'index')
enrondf = enrondf.replace('NaN', np.nan)

### Explore data
print 'Number of executives:', len(enrondf)
print 'Number of POIs:', len(enrondf.loc[enrondf.poi == True, 'poi'])
print 'Number of features:', enrondf.shape[1]

enrondf.describe()

### Task 2: Remove outliers

enrondf[enrondf.loc[:,enrondf.columns!='poi'].isnull().all(axis=1)]

### Drop LOCKHART EUGENE E as all the fields are NaN

enrondf.drop('LOCKHART EUGENE E', inplace = True)

#check data: summing payments features and compare with total_payments
payments = ['salary',
            'bonus', 
            'long_term_incentive', 
            'deferred_income', 
            'deferral_payments',
            'loan_advances', 
            'other',
            'expenses', 
            'director_fees']

enrondf[enrondf[payments].sum(axis='columns') != enrondf.total_payments]

stock_value = ['exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred']

enrondf[enrondf[stock_value].sum(axis='columns') != enrondf.total_stock_value]

#Adjusting the executives' payment and stock data

enrondf.loc['BELFER ROBERT','total_payments'] = 3285
enrondf.loc['BELFER ROBERT','deferral_payments'] = 0
enrondf.loc['BELFER ROBERT','restricted_stock'] = 44093
enrondf.loc['BELFER ROBERT','restricted_stock_deferred'] = -44093
enrondf.loc['BELFER ROBERT','total_stock_value'] = 0
enrondf.loc['BELFER ROBERT','director_fees'] = 102500
enrondf.loc['BELFER ROBERT','deferred_income'] = -102500
enrondf.loc['BELFER ROBERT','exercised_stock_options'] = 0
enrondf.loc['BELFER ROBERT','expenses'] = 3285
enrondf.loc['BELFER ROBERT']
enrondf.loc['BHATNAGAR SANJAY','expenses'] = 137864
enrondf.loc['BHATNAGAR SANJAY','total_payments'] = 137864
enrondf.loc['BHATNAGAR SANJAY','exercised_stock_options'] = 1.54563e+07
enrondf.loc['BHATNAGAR SANJAY','restricted_stock'] = 2.60449e+06
enrondf.loc['BHATNAGAR SANJAY','restricted_stock_deferred'] = -2.60449e+06
enrondf.loc['BHATNAGAR SANJAY','other'] = 0
enrondf.loc['BHATNAGAR SANJAY','director_fees'] = 0
enrondf.loc['BHATNAGAR SANJAY','total_stock_value'] = 1.54563e+07
enrondf.loc['BHATNAGAR SANJAY']

### Plot a scatterplot
pois = enrondf[enrondf.poi]
plt.scatter(pois["salary"], pois["bonus"], c='red');
nonpois = enrondf[~enrondf.poi]
plt.scatter(nonpois["salary"], nonpois["bonus"], c='blue');
plt.xlabel("Salary")
plt.ylabel("Bonus")

plt.legend(["POI","Not a POI"])

### Identifying outliers by looking at the top earners
enrondf[enrondf.loc[:,"salary"]>2000000]

#Drop Total as it looks like the total of everybody's salaries
enrondf = enrondf.drop("TOTAL")

enrondf.drop('THE TRAVEL AGENCY IN THE PARK', inplace=True)

### Plot again 
pois = enrondf[enrondf.poi]
plt.scatter(pois["salary"], pois["bonus"], c='red');
nonpois = enrondf[~enrondf.poi]
plt.scatter(nonpois["salary"], nonpois["bonus"], c='blue');
plt.xlabel("Salary")
plt.ylabel("Bonus")

plt.legend(["POI","Not a POI"])


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
enrondf['fraction_from_poi'] = enrondf['from_poi_to_this_person'] / enrondf['to_messages']
enrondf['fraction_to_poi'] = enrondf['from_this_person_to_poi'] / enrondf['from_messages']
enrondf['fraction_shared_receipt'] = enrondf['shared_receipt_with_poi'] / enrondf['to_messages']

features_list.append('fraction_from_poi')
features_list.append('fraction_to_poi')
features_list.append('fraction_shared_receipt')

# Replacing 'NaN' in financial features with 0
enrondf = enrondf.fillna(0)

# back into dict
data_dict = enrondf.to_dict(orient='index')

my_dataset = data_dict

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import tester

def set_classifier(x):
    # switch statement Python replacement - http://stackoverflow.com/a/103081
    return {
        'rfc': RandomForestClassifier(),
        'dtr': DecisionTreeClassifier(),
        'gnb': GaussianNB(),
        'kn': KNeighborsClassifier(2),
        'ada': AdaBoostClassifier(),
        'xtra': ExtraTreesClassifier(),
        'km': KMeans(n_clusters=2, random_state=0)
    }.get(x)

#Test for all the algorithms defined earlier
for classifier in ['rfc','dtr','gnb','kn','ada','xtra','km']:
    clf = set_classifier(classifier)
    tester.dump_classifier_and_data(clf, my_dataset, features_list)
    tester.main();
    
    print "====================================================================="

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

features_list = ['fraction_to_poi','total_stock_value', 'exercised_stock_options', 'bonus', 'salary', 'fraction_shared_receipt',  'deferred_income', 'shared_receipt_with_poi', 'long_term_incentive', 'total_payments', 'restricted_stock', 'loan_advances', 'from_poi_to_this_person', 'expenses', 'other']

tree_clf = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=20,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

# Use GridSearchCV to automate the process of finding the optimal number of features
# To speed up the process of verification, the end result that I recieved from my work has been pasted directly
# Final script in Enron_ML.ipynb
#tree_clf.best_params_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(tree_clf, my_dataset, features_list)
