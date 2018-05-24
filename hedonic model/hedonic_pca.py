
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
mlinput=postimpute_pca
train_set, test_set = train_test_split(mlinput, test_size=0.3, random_state=42)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in split.split(mlinput, mlinput["city_final"]):
    strat_train_set_wo_MGR1 = mlinput.loc[train_index]
    strat_test_set_wo_MGR1 = mlinput.loc[test_index]

del(train_index,test_index,split)






param_grid = [
{'learning_rate': [0.01,0.03,0.05,0.07,0.09], 'n_estimators': [200,300,400,500], 'max_depth': [2,3,4], 'subsample': [0.33,0.67,1], 'max_features':["auto","sqrt","log2"], 'loss':["ls", "lad", "huber"] },
]

gbrt = GradientBoostingRegressor()
grid_search = GridSearchCV(gbrt, param_grid, cv=5)


ind = strat_train_set_wo_MGR1 .copy()
# Removing column
list_drop = ['address_final','zipcode_final','city_final','state_final','rent_final']
ind.drop(list_drop, axis=1, inplace=True)
dep = strat_train_set_wo_MGR1 ['rent_final'].copy()
grid_search.fit(ind, dep)
grid_search.best_params_
grid_search.best_estimator_
grid_search.get_params()
grid_search.get_params().keys()



gbm2 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.05, loss='huber', max_depth=4,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=300,
             presort='auto', random_state=None, subsample=0.67, verbose=0,
             warm_start=False)
gbm2.fit(ind, dep)

#Cross-validated Train errors
rent_predictions = gbm2.predict(ind)
rf1_mse = mean_squared_error(dep, rent_predictions)
rf1_rmse = numpy.sqrt(rf1_mse)
rf1_rmse
#392.864
sklearn.metrics.r2_score(dep, rent_predictions)*100
#89.435
def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100

def mean_percentage_error(y_true, y_pred): 
    return numpy.mean((y_true - y_pred) / y_true) * 100

mean_absolute_percentage_error(dep, rent_predictions)
#9.89
mean_percentage_error(dep, rent_predictions)
#-1.44

#test error

ind_test = strat_test_set_wo_MGR1 .copy()
# Removing column
list_drop = ['address_final','zipcode_final','city_final','state_final','rent_final']
ind_test.drop(list_drop, axis=1, inplace=True)
dep_test = strat_test_set_wo_MGR1 ['rent_final'].copy()


rent_predictions_test = gbm2.predict(ind_test)
rf1_mse_test = mean_squared_error(dep_test, rent_predictions_test)
rf1_rmse_test = numpy.sqrt(rf1_mse_test)
rf1_rmse_test
#640.185
sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#75.89
mean_absolute_percentage_error(dep_test, rent_predictions_test)
#19.409
mean_percentage_error(dep_test, rent_predictions_test)
#-6.34

#Variable importance
feature_importances = gbm2.feature_importances_
feature_importances
attributes = ind.columns
pandas.DataFrame(sorted(zip(feature_importances, attributes), reverse=True)).to_csv('/Users/aditilakra/Desktop/capstone/hedonic model/stoachastic_gbm_varimp_ratioofzipcoderent2.csv')

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
#                 len
                MAPE
MAPE_bins           
(0, 10]        619.0
(10, 20]       240.0
(20, 30]        69.0
(30, 50]        30.0
(50, 100]       11.0
(100, 100000]    1.0

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])

#                  len
                   MPE
MPE_bins              
(-100000, -100]    1.0
(-100, -50]        7.0
(-50, -30]        17.0
(-30, -20]        47.0
(-20, -10]       128.0
(-10, 0]         311.0
(0, 10]          308.0
(10, 20]         112.0
(20, 30]          22.0
(30, 50]          13.0
(50, 100]          4.0
(100, 100000]      NaN

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
0        (0, 500]  65.834012      4  65.834012
1     (500, 1000]  12.275433    235   9.620164
2    (1000, 2000]   8.040671    505  -0.535262
3    (2000, 3000]   8.485752    113  -2.307853
4    (3000, 4000]  10.806200     50  -5.430450
5  (4000, 100000]  14.023705     63 -12.875321
    
    
 # precision
         Rent_bins       MAPE  count       MPE
0        (0, 500]   4.605207      1 -4.605207
1     (500, 1000]   9.915530    192  4.169709
2    (1000, 2000]   9.739175    564 -0.963345
3    (2000, 3000]  10.049742    107 -4.068250
4    (3000, 4000]  12.233189     55 -6.493860
5  (4000, 100000]   8.750661     51 -6.808803
    
del(a,b,c,d)

a = train_eb.groupby('city_final').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MPE']

d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)

#     city_final       MAPE  count       MPE
0       Austin  10.507271    386 -2.019325
1  Chattanooga   7.430974     44  1.269823
2      Chicago   9.486328    287 -2.378421
3     Columbia  10.936145     94 -1.349942
4       Dayton  13.483123     57  2.535464
5       Eugene  11.096742     44  0.788342
6      Newyork  10.299951     58 -6.502114

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

#                  len
                MAPE
MAPE_bins           
(0, 10]        165.0
(10, 20]       104.0
(20, 30]        63.0
(30, 50]        55.0
(50, 100]       26.0
(100, 100000]    3.0

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])

#                  len
                  MPE
MPE_bins             
(-100000, -100]   3.0
(-100, -50]      20.0
(-50, -30]       39.0
(-30, -20]       37.0
(-20, -10]       62.0
(-10, 0]         74.0
(0, 10]          91.0
(10, 20]         42.0
(20, 30]         26.0
(30, 50]         16.0
(50, 100]         6.0
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

# 
        Rent_bins       MAPE  count        MPE
0        (0, 500]  42.088205      3  23.960940
1     (500, 1000]  23.471501    114  20.680971
2    (1000, 2000]  16.019313    201   3.785241
3    (2000, 3000]  20.801487     52  -0.708724
4    (3000, 4000]  12.971560     17  -4.780149
5  (4000, 100000]  26.222776     29 -24.143383
    
    
    
#precision
        Rent_bins       MAPE  count        MPE
0        (0, 500]  25.899240      1 -25.899240
1     (500, 1000]  15.580590     85  -0.817714
2    (1000, 2000]  19.540377    232   0.213687
3    (2000, 3000]  21.112789     51  -1.764013
4    (3000, 4000]  24.486262     25  -0.204198
5  (4000, 100000]  20.529525     22 -13.686727
    
    
    
    
del(a,b,c,d)

a = train_eb.groupby('city_final').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MPE']

d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)

#     city_final       MAPE  count        MPE
0       Austin  18.177660    166  -1.010604
1  Chattanooga  20.711734     19   5.332569
2      Chicago  19.948002    123   5.018366
3     Columbia  15.078039     40   6.432423
4       Dayton  24.563071     24  11.380207
5       Eugene  30.261371     19 -17.080527
6      Newyork  21.289455     25 -17.081118

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
# 6.60%%

median_percentage_error(dep, rent_predictions)
#0.71%

#test error

median_absolute_percentage_error(dep_test, rent_predictions_test)
# 13.43%

median_percentage_error(dep_test, rent_predictions_test)
#4.21%

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
 
        Rent_bins      MdAPE  count       MdPE
0        (0, 500]  56.275648      4  56.275648
1     (500, 1000]   9.031625    235   7.378587
2    (1000, 2000]   6.289808    505  -0.776641
3    (2000, 3000]   5.818146    113  -1.219317
4    (3000, 4000]   5.220792     50   0.730177
5  (4000, 100000]   4.827420     63  -4.827420
    
 # precsiom
        Rent_bins     MdAPE  count      MdPE
0        (0, 500]  4.605207      1 -4.6052074
1     (500, 1000]  6.684789    192  3.370830
2    (1000, 2000]  7.370452    564  0.281550
3    (2000, 3000]  5.792652    107  0.747583
4    (3000, 4000]  5.443491     55  0.462335
5  (4000, 100000]  4.394139     51 -2.799256
    
    
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
0       Austin   7.163589    386  0.046194
1  Chattanooga   4.675010     44  2.527903
2      Chicago   6.115022    287  0.416086
3     Columbia   6.355629     94  1.491637
4       Dayton  11.761876     57  6.456890
5       Eugene   6.624962     44  0.327233
6      Newyork   3.989195     58 -0.554292
del(a,b,c,d)

#Error buckets - by MdAPE/MdPE, rent range and city - test

a=strat_test_set_wo_MGR1.iloc[:,3].values
b=strat_test_set_wo_MGR1.iloc[:,4].values
train_eb = pandas.DataFrame({'rent_final': a, 'city_final': b, 'rent_predictions': rent_predictions_test}, columns=['rent_final', 'city_final', 'rent_predictions'])
del(a,b)

bins = [0, 500, 1000, 2000, 3000, 4000, 100000]
train_eb['Rent_bins'] = pandas.cut(train_eb['rent_predictions'], bins)

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
        Rent_bins      MdAPE  count       MdPE
0        (0, 500]  27.190897      3  26.091700
1     (500, 1000]  15.404264    114  14.303857
2    (1000, 2000]  10.925969    201   3.187103
3    (2000, 3000]  17.609767     52  -3.945886
4    (3000, 4000]   9.549451     17   0.768977
5  (4000, 100000]  22.258605     29 -18.397067

# precision
        Rent_bins      MdAPE  count       MdPE
0        (0, 500]  25.899240      1 -25.899240
1     (500, 1000]  11.808210     85   2.330442
2    (1000, 2000]  14.011147    232   4.463225
3    (2000, 3000]  15.616434     51   3.093136
4    (3000, 4000]  26.080207     25   6.096654
5  (4000, 100000]  17.008352     22  -9.666119
del(a,b,c,d)

a = train_eb.groupby('city_final').apply(median_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MdAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(median_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MdPE']
d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MdPE']], how='left', left_index=True, right_index=True)

#     city_final      MdAPE  count       MdPE
0       Austin  11.980767    166   2.994453
1  Chattanooga  10.833883     19   4.915972
2      Chicago  14.915629    123   7.312736
3     Columbia  12.534918     40   9.330210
4       Dayton  23.087584     24  16.189904
5       Eugene  17.178720     19  -0.369281
6      Newyork  14.815066     25 -12.449429


del(a,b,c,d)


