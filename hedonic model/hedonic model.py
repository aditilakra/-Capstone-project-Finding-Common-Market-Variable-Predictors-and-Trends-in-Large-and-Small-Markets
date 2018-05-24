import pandas
import numpy
import os
import csv
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import sklearn

#

postimpute_finaldata = pandas.read_csv('/Users/aditilakra/Desktop/capstone/Dataaggregation/PostImpute_finaldata.csv')
postimpute_finaldata.drop('Unnamed: 0', axis=1, inplace=True)


postimpute_pca = pandas.read_csv('/Users/aditilakra/Desktop/capstone/Dataaggregation/PostImpute_pca.csv')
postimpute_pca.drop('Unnamed: 0', axis=1, inplace=True)

#Split data into train and test (30%) 
#Random Forests for variable importance & GBM for rental predictions on weighted rents
mlinput=postimpute_finaldata
train_set, test_set = train_test_split(mlinput, test_size=0.3, random_state=42)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in split.split(mlinput, mlinput["city_final"]):
    strat_train_set_wo_MGR1 = mlinput.loc[train_index]
    strat_test_set_wo_MGR1 = mlinput.loc[test_index]

del(train_index,test_index,split)

postimpute_finaldata["city_final"].value_counts()/len(postimpute_finaldata)
#Austin         0.398268
#Chicago        0.295815
#Columbia       0.096681
#Newyork        0.059885
#Dayton         0.058442
#Eugene         0.045455
#Chattanooga    0.045455
train_set["city_final"].value_counts()/len(train_set)
test_set["city_final"].value_counts()/len(test_set)
(strat_train_set_wo_MGR1["city_final"].value_counts()/len(strat_train_set_wo_MGR1) - strat_test_set_wo_MGR1["city_final"].value_counts()/len(strat_test_set_wo_MGR1))/postimpute_finaldata["city_final"].value_counts()/len(postimpute_finaldata)*100
(train_set["city_final"].value_counts()/len(train_set) - test_set["city_final"].value_counts()/len(test_set))/postimpute_finaldata["city_final"].value_counts()/len(postimpute_finaldata)*100
#the difference in city-wise proportions is less in the stratified split
Austin        -0.000002
Chattanooga   -0.000004
Chicago       -0.000002
Columbia       0.000010
Dayton         0.000004
Eugene         0.000019
Newyork       -0.000003




param_grid = [
{'learning_rate': [0.01,0.03,0.05,0.07,0.09], 'n_estimators': [200,300,400,500], 'max_depth': [2,3,4], 'subsample': [0.33,0.67,1], 'max_features':["auto","sqrt","log2"], 'loss':["ls", "lad", "huber"] },
]


gbrt = GradientBoostingRegressor()
grid_search = GridSearchCV(gbrt, param_grid, cv=5)
ind = strat_train_set_wo_MGR1.copy()
# Removing column
list_drop = ['address_final','zipcode_final','city_final','state_final','rent_final']
ind.drop(list_drop, axis=1, inplace=True)
dep = strat_train_set_wo_MGR1['rent_final'].copy()


grid_search.fit(ind, dep)
grid_search.best_params_
grid_search.best_estimator_



gbm1 =  GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.05, loss='huber', max_depth=4,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=300,
             presort='auto', random_state=None, subsample=0.33, verbose=0,
             warm_start=False)
gbm1.fit(ind, dep)

#Cross-validated Train errors
rent_predictions = gbm1.predict(ind)
rf1_mse = mean_squared_error(dep, rent_predictions)
rf1_rmse = numpy.sqrt(rf1_mse)
rf1_rmse
#406.41
sklearn.metrics.r2_score(dep, rent_predictions)*100
#88.69
def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100

def mean_percentage_error(y_true, y_pred): 
    return numpy.mean((y_true - y_pred) / y_true) * 100

mean_absolute_percentage_error(dep, rent_predictions)
#12.05
mean_percentage_error(dep, rent_predictions)
#-2.50

#test error

ind_test = strat_test_set_wo_MGR1.copy()
# Removing column
list_drop = ['address_final','zipcode_final','city_final','state_final','rent_final']
ind_test.drop(list_drop, axis=1, inplace=True)
dep_test = strat_test_set_wo_MGR1['rent_final'].copy()


rent_predictions_test = gbm1.predict(ind_test)
rf1_mse_test = mean_squared_error(dep_test, rent_predictions_test)
rf1_rmse_test = numpy.sqrt(rf1_mse_test)
rf1_rmse_test
#607.61
sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#78.28
mean_absolute_percentage_error(dep_test, rent_predictions_test)
#19.51
mean_percentage_error(dep_test, rent_predictions_test)
#-6.353

#Variable importance
feature_importances = gbm1.feature_importances_
feature_importances
attributes = ind.columns
pandas.DataFrame(sorted(zip(feature_importances, attributes), reverse=True)).to_csv('/Users/aditilakra/Desktop/capstone/hedonic model/stoachastic_gbm_varimp_ratioofzipcoderent1.csv')

#Error buckets - by MAPE/MPE, rent range and city - train

a=strat_train_set_wo_MGR1.iloc[:,3].values
b=strat_train_set_wo_MGR1.iloc[:,4].values
train_eb = pandas.DataFrame({'rent_final': a, 'city_final': b, 'rent_predictions': rent_predictions}, columns=['rent_final', 'city_final', 'rent_predictions'])
del(a,b)
train_eb['MAPE'] = (numpy.abs(train_eb['rent_final'] - train_eb['rent_predictions']) / train_eb['rent_final']) * 100
train_eb['MPE'] = ((train_eb['rent_final'] - train_eb['rent_predictions']) / train_eb['rent_final']) * 100
bins = [0, 10, 20, 30, 50, 100, 100000]
train_eb['MAPE_bins'] = pandas.cut(train_eb['MAPE'], bins)
bins = [-100000, -100, -50, -30, -20, -10,  0, 10, 20, 30, 50, 100, 100000]
train_eb['MPE_bins'] = pandas.cut(train_eb['MPE'], bins)
pandas.pivot_table(train_eb,index=["MAPE_bins"],values=["MAPE"],aggfunc=[len])
#
                len
                MAPE
MAPE_bins           
(0, 10]        544.0
(10, 20]       279.0
(20, 30]        92.0
(30, 50]        30.0
(50, 100]       21.0
(100, 100000]    4.0

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])

#
                    len
                   MPE
MPE_bins              
(-100000, -100]    4.0
(-100, -50]       15.0
(-50, -30]        20.0
(-30, -20]        61.0
(-20, -10]       147.0
(-10, 0]         252.0
(0, 10]          292.0
(10, 20]         132.0
(20, 30]          31.0
(30, 50]          10.0
(50, 100]          6.0
(100, 100000]    

bins = [0, 500, 1000, 2000, 3000, 4000, 100000]
train_eb['Rent_bins'] = pandas.cut(train_eb['rent_final'], bins)

def mean_absolute_percentage_error_df(df,y_true, y_pred): 
    return numpy.sum(numpy.abs(df[y_pred] - df[y_true])) / numpy.sum(df[y_true]) * 100

def mean_percentage_error_df(df, y_true, y_pred): 
    return numpy.sum((df[y_pred] - df[y_true])) / numpy.sum(df[y_true]) * 100

a = train_eb.groupby('Rent_bins').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['Rent_bins','MAPE']
b = train_eb.groupby('Rent_bins').size().reset_index()
b.columns = ['Rent_bins','count']
c = train_eb.groupby('Rent_bins').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['Rent_bins','MPE']

d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)

#        Rent_bins        MAPE  count         MPE
0        (0, 500]  102.683593      4  102.683593
1     (500, 1000]   14.701376    235   11.068839
2    (1000, 2000]   10.127923    505    0.036143
3    (2000, 3000]   10.054807    113   -1.892355
4    (3000, 4000]   11.356072     50   -4.643045
5  (4000, 100000]   14.482457     63  -11.866154
    
    
del(a,b,c,d)

a = train_eb.groupby('city_final').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MPE']

d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)

#  
    city_final       MAPE  count       MPE
0       Austin  12.606277    386 -1.852527
1  Chattanooga   9.122462     44 -1.275774
2      Chicago  11.631661    287 -1.843735
3     Columbia  13.220608     94 -2.384759
4       Dayton  12.690111     57  2.522807
5       Eugene  15.602468     44  1.466624
6      Newyork   9.384181     58 -2.568911

del(a,b,c,d)

#Error buckets - by MAPE/MPE, rent range and city - test

a=strat_test_set_wo_MGR1.iloc[:,3].values
b=strat_test_set_wo_MGR1.iloc[:,4].values
train_eb = pandas.DataFrame({'rent_final': a, 'city_final': b, 'rent_predictions': rent_predictions_test}, columns=['rent_final', 'city_final', 'rent_predictions'])
del(a,b)
train_eb['MAPE'] = (numpy.abs(train_eb['rent_final'] - train_eb['rent_predictions']) / train_eb['rent_final']) * 100
train_eb['MPE'] = ((train_eb['rent_final'] - train_eb['rent_predictions']) / train_eb['rent_final']) * 100
bins = [0, 10, 20, 30, 50, 100, 100000]
train_eb['MAPE_bins'] = pandas.cut(train_eb['MAPE'], bins)
bins = [-100000, -100, -50, -30, -20, -10,  0, 10, 20, 30, 50, 100, 100000]
train_eb['MPE_bins'] = pandas.cut(train_eb['MPE'], bins)
pandas.pivot_table(train_eb,index=["MAPE_bins"],values=["MAPE"],aggfunc=[len])

#                 len
                MAPE
MAPE_bins           
(0, 10]        149.0
(10, 20]       126.0
(20, 30]        65.0
(30, 50]        46.0
(50, 100]       26.0
(100, 100000]    4.0

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])

#                  len
                  MPE
MPE_bins             
(-100000, -100]   4.0
(-100, -50]      21.0
(-50, -30]       27.0
(-30, -20]       40.0
(-20, -10]       73.0
(-10, 0]         81.0
(0, 10]          68.0
(10, 20]         53.0
(20, 30]         25.0
(30, 50]         19.0
(50, 100]         5.0
(100, 100000]     NaN

bins = [0, 500, 1000, 2000, 3000, 4000, 100000]
train_eb['Rent_bins'] = pandas.cut(train_eb['rent_final'], bins)

def mean_absolute_percentage_error_df(df,y_true, y_pred): 
    return numpy.sum(numpy.abs(df[y_pred] - df[y_true])) / numpy.sum(df[y_true]) * 100

def mean_percentage_error_df(df, y_true, y_pred): 
    return numpy.sum((df[y_pred] - df[y_true])) / numpy.sum(df[y_true]) * 100

a = train_eb.groupby('Rent_bins').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['Rent_bins','MAPE']
b = train_eb.groupby('Rent_bins').size().reset_index()
b.columns = ['Rent_bins','count']
c = train_eb.groupby('Rent_bins').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['Rent_bins','MPE']

d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)

#        Rent_bins       MAPE  count        MPE
0        (0, 500]  67.880470      3  67.880470
1     (500, 1000]  21.335447    114  18.587020
2    (1000, 2000]  16.713646    201   3.797110
3    (2000, 3000]  23.005136     52  -1.712794
4    (3000, 4000]  14.469449     17  -3.283081
5  (4000, 100000]  23.469749     29 -19.772853
    
del(a,b,c,d)

a = train_eb.groupby('city_final').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MPE']

d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)

#    city_final       MAPE  count        MPE
0       Austin  18.845281    166  -1.531647
1  Chattanooga  20.228212     19   2.081663
2      Chicago  20.953156    123   5.357137
3     Columbia  16.064345     40   5.185520
4       Dayton  22.105199     24   6.650759
5       Eugene  27.605542     19 -17.673199
6      Newyork  18.587823     25 -11.394511


del(a,b,c,d)

joblib.dump(gbm8, "gbm8.pkl")
#gbm8 = joblib.load("gbm8.pkl")

####################################
####################################
####################################

#Median based errors

#Cross-validated Train errors

def median_absolute_percentage_error(y_true, y_pred): 
    return numpy.median((numpy.abs(y_pred - y_true)/y_true)*100)

def median_percentage_error(y_true, y_pred): 
    return numpy.median(((y_pred - y_true) / y_true) * 100)

median_absolute_percentage_error(dep, rent_predictions)
# 8.54%

median_percentage_error(dep, rent_predictions)
#0.45%

#test error

median_absolute_percentage_error(dep_test, rent_predictions_test)
# 14.34%

median_percentage_error(dep_test, rent_predictions_test)
#4.30%

#Error buckets - by MdAPE/MdPE, rent range and city - train
a=strat_train_set_wo_MGR1.iloc[:,3].values
b=strat_train_set_wo_MGR1.iloc[:,4].values
train_eb = pandas.DataFrame({'rent_final': a, 'city_final': b, 'rent_predictions': rent_predictions}, columns=['rent_final', 'city_final', 'rent_predictions'])
del(a,b)

bins = [0, 500, 1000, 2000, 3000, 4000, 100000]
train_eb['Rent_bins'] = pandas.cut(train_eb['rent_final'], bins)

def median_absolute_percentage_error_df(df,y_true, y_pred): 
    return numpy.median((numpy.abs(df[y_pred] - df[y_true]) / df[y_true]) * 100)

def median_percentage_error_df(df, y_true, y_pred): 
    return numpy.median(((df[y_pred] - df[y_true]) / df[y_true]) * 100)

a = train_eb.groupby('Rent_bins').apply(median_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['Rent_bins','MdAPE']
b = train_eb.groupby('Rent_bins').size().reset_index()
b.columns = ['Rent_bins','count']
c = train_eb.groupby('Rent_bins').apply(median_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['Rent_bins','MdPE']
d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MdPE']], how='left', left_index=True, right_index=True)

#

        Rent_bins       MdAPE  count        MdPE
0        (0, 500]  105.070386      4  105.070386
1     (500, 1000]    9.636681    235    7.589477
2    (1000, 2000]    8.479047    505   -0.799231
3    (2000, 3000]    8.091251    113   -2.134392
4    (3000, 4000]    6.575575     50   -0.967811
5  (4000, 100000]    6.717651     63   -4.794779

del(a,b,c,d)

a = train_eb.groupby('city_final').apply(median_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MdAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(median_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MdPE']
d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MdPE']], how='left', left_index=True, right_index=True)

#    city_final      MdAPE  count      MdPE
0       Austin   8.477241    386  0.952053
1  Chattanooga   7.310070     44  0.599838
2      Chicago   9.179021    287 -0.329769
3     Columbia   9.181184     94 -1.248619
4       Dayton   8.812614     57  3.749416
5       Eugene  10.622100     44  0.343967
6      Newyork   5.230391     58 -0.187233

del(a,b,c,d)

#Error buckets - by MdAPE/MdPE, rent range and city - test

a=strat_test_set_wo_MGR1.iloc[:,3].values
b=strat_test_set_wo_MGR1.iloc[:,4].values
train_eb = pandas.DataFrame({'rent_final': a, 'city_final': b, 'rent_predictions': rent_predictions_test}, columns=['rent_final', 'city_final', 'rent_predictions'])
del(a,b)

bins = [0, 500, 1000, 2000, 3000, 4000, 100000]
train_eb['Rent_bins'] = pandas.cut(train_eb['rent_final'], bins)

def median_absolute_percentage_error_df(df,y_true, y_pred): 
    return numpy.median((numpy.abs(df[y_pred] - df[y_true]) / df[y_true]) * 100)

def median_percentage_error_df(df, y_true, y_pred): 
    return numpy.median(((df[y_pred] - df[y_true]) / df[y_true]) * 100)

a = train_eb.groupby('Rent_bins').apply(median_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['Rent_bins','MdAPE']
b = train_eb.groupby('Rent_bins').size().reset_index()
b.columns = ['Rent_bins','count']
c = train_eb.groupby('Rent_bins').apply(median_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['Rent_bins','MdPE']
d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MdPE']], how='left', left_index=True, right_index=True)

        Rent_bins      MdAPE  count       MdPE
0        (0, 500]  57.144924      3  57.144924
1     (500, 1000]  13.241761    114  12.951616
2    (1000, 2000]  13.761046    201   2.488461
3    (2000, 3000]  21.807928     52  -4.460135
4    (3000, 4000]  14.260970     17  -0.837563
5  (4000, 100000]  17.473506     29 -13.755571
    
    
del(a,b,c,d)

a = train_eb.groupby('city_final').apply(median_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MdAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(median_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MdPE']
d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MdPE']], how='left', left_index=True, right_index=True)

#    city_final      MdAPE  count       MdPE
0       Austin  14.297778    166   4.512457
1  Chattanooga  15.592141     19  -2.646848
2      Chicago  15.994021    123   8.047309
3     Columbia  12.157794     40   7.729186
4       Dayton  14.419226     24  10.461145
5       Eugene  11.922341     19  -1.217664
6      Newyork  13.755571     25  -6.027441

del(a,b,c,d)


