import pandas
import numpy as np
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
import math


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
{'learning_rate': [0.03,0.05,0.07], 'n_estimators': [200,300,400], 'max_depth': [3,4], 'subsample': [0.33,0.67], 'max_features':["sqrt","log2"], 'loss':["ls","huber"] },
]

gbrt = GradientBoostingRegressor()
grid_search = GridSearchCV(gbrt, param_grid, cv=5)


ind = strat_train_set_wo_MGR1 .copy()

# Removing column
list_drop = ['address_final','zipcode_final','city_final','state_final','rent_final']
ind.drop(list_drop, axis=1, inplace=True)
dep = strat_train_set_wo_MGR1 ['rent_final'].copy()


log=lambda x: math.log(x)
exp = lambda x: math.exp (x)

dep.new= dep.apply(lambda x: math.log (x))
dep.new.min()
grid_search.fit(ind, dep.new)
grid_search.best_params_
grid_search.best_estimator_


gbm2 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.07, loss='huber', max_depth=4,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=300,
             presort='auto', random_state=None, subsample=0.67, verbose=0,
             warm_start=False)
gbm2.fit(ind, dep.new)

#Cross-validated Train errors
pred = gbm2.predict(ind)
rent_predictions = np.exp(pred)
rf1_mse = mean_squared_error(dep, rent_predictions)
rf1_rmse = numpy.sqrt(rf1_mse)
rf1_rmse
#320.30
sklearn.metrics.r2_score(dep, rent_predictions)*100
#92.97
def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100

def mean_percentage_error(y_true, y_pred): 
    return numpy.mean((y_true - y_pred) / y_true) * 100

mean_absolute_percentage_error(dep, rent_predictions)
#7.08
mean_percentage_error(dep, rent_predictions)
#-0.17

#test error

ind_test = strat_test_set_wo_MGR1 .copy()
# Removing column
list_drop = ['address_final','zipcode_final','city_final','state_final','rent_final']
ind_test.drop(list_drop, axis=1, inplace=True)
dep_test = strat_test_set_wo_MGR1 ['rent_final'].copy()
log=lambda x: math.log (x)
exp = lambda x: math.exp (x)

dep.new= dep.apply(lambda x: math.log (x))
dep.new.min()



#Cross-validated Train errors
pred_test=gbm2.predict(ind_test)
rent_predictions_test = np.exp(pred_test)
#rent_predictions_test = gbm2.predict(ind_test)
rf1_mse_test = mean_squared_error(dep_test, rent_predictions_test)
rf1_rmse_test = numpy.sqrt(rf1_mse_test)
rf1_rmse_test
#644.90
sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#75.54
mean_absolute_percentage_error(dep_test, rent_predictions_test)
#17.90
mean_percentage_error(dep_test, rent_predictions_test)
#-2.69

#Variable importance
feature_importances = gbm2.feature_importances_
feature_importances
attributes = ind.columns
pandas.DataFrame(sorted(zip(feature_importances, attributes), reverse=True)).to_csv('/Users/aditilakra/Desktop/capstone/hedonic model/stoachastic_gbm_varimp_ratioofzipcoderent_log.csv')

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
#                   len
                  MAPE
MAPE_bins           
(0, 10]        758.0
(10, 20]       160.0
(20, 30]        31.0
(30, 50]        14.0
(50, 100]        7.0
(100, 100000]    NaN

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])

#                   len
                   MPE
MPE_bins              
(-100000, -100]    NaN
(-100, -50]        3.0
(-50, -30]         6.0
(-30, -20]        15.0
(-20, -10]        86.0
(-10, 0]         376.0
(0, 10]          382.0
(10, 20]          74.0
(20, 30]          16.0
(30, 50]           8.0
(50, 100]          4.0
(100, 100000]      NaN

bins = [0, 500, 1000, 2000, 3000, 4000, 100000]
train_eb['Rent_bins'] = pandas.cut(train_eb['rent_predictions'], bins)

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

#         Rent_bins       MAPE  count        MPE
0        (0, 500]  38.680493      4  38.680493
1     (500, 1000]   7.338851    235   5.100282
2    (1000, 2000]   6.253385    505  -1.122667
3    (2000, 3000]   6.605902    113  -1.882948
4    (3000, 4000]   9.782791     50  -4.680415
5  (4000, 100000]  10.582562     63  -8.400745
    
 ### precision
          Rent_bins      MAPE  count       MPE
0        (0, 500]  4.997207      2  4.358459
1     (500, 1000]  6.194752    210  2.408348
2    (1000, 2000]  7.741644    548 -2.097707
3    (2000, 3000]  7.420368    104 -2.303690
4    (3000, 4000]  8.721886     52 -3.492978
5  (4000, 100000]  8.976705     54 -6.421708
    
del(a,b,c,d)

a = train_eb.groupby('city_final').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MPE']

d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)

#     city_final      MAPE  count       MPE
0       Austin  8.361776    386 -2.498973
1  Chattanooga  5.029039     44 -0.665429
2      Chicago  7.236364    287 -1.790618
3     Columbia  7.987460     94 -3.346977
4       Dayton  8.498930     57 -1.083099
5       Eugene  7.842396     44 -0.806109
6      Newyork  7.564236     58 -3.431824

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
(0, 10]        166.0
(10, 20]       114.0
(20, 30]        69.0
(30, 50]        42.0
(50, 100]       24.0
(100, 100000]    1.0

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])

#                   len
                  MPE
MPE_bins             
(-100000, -100]   1.0
(-100, -50]      17.0
(-50, -30]       22.0
(-30, -20]       36.0
(-20, -10]       59.0
(-10, 0]         83.0
(0, 10]          83.0
(10, 20]         55.0
(20, 30]         33.0
(30, 50]         20.0
(50, 100]         7.0
(100, 100000]     NaN

bins = [0, 500, 1000, 2000, 3000, 4000, 100000]
train_eb['Rent_bins'] = pandas.cut(train_eb['rent_predictions'], bins)

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

#         Rent_bins       MAPE  count        MPE
0        (0, 500]  29.366250      3  24.580647
1     (500, 1000]  19.442858    114  16.441487
2    (1000, 2000]  14.514062    201  -0.842596
3    (2000, 3000]  22.997627     52  -2.265233
4    (3000, 4000]  15.315795     17  -3.588914
5  (4000, 100000]  28.293078     29 -25.298871
    
# precision    
            Rent_bins       MAPE  count       MPE
0        (0, 500]   6.125145      2 -1.904414
1     (500, 1000]  16.391180     93 -2.793242
2    (1000, 2000]  18.326069    231 -2.904373
3    (2000, 3000]  17.262720     43 -4.373862
4    (3000, 4000]  23.845059     21 -4.073924
5  (4000, 100000]  16.431392     26 -2.742510
    
del(a,b,c,d)

a = train_eb.groupby('city_final').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MPE']

d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)

#         city_final       MAPE  count        MPE
0       Austin  16.887874    166  -3.754530
1  Chattanooga  17.446533     19   0.031017
2      Chicago  20.286268    123   1.408104
3     Columbia  13.376418     40   3.332339
4       Dayton  23.405480     24   8.272586
5       Eugene  27.803701     19 -18.388012
6      Newyork  23.914189     25 -18.138841


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
# 4.85

median_percentage_error(dep, rent_predictions)
#0.033

#test error

median_absolute_percentage_error(dep_test, rent_predictions_test)
# 12.48

median_percentage_error(dep_test, rent_predictions_test)
#1.17

#Error buckets - by MdAPE/MdPE, rent range and city - train
a=strat_train_set_wo_MGR1.iloc[:,3].values
b=strat_train_set_wo_MGR1.iloc[:,4].values
train_eb = pandas.DataFrame({'rent_final': a, 'city_final': b, 'rent_predictions': rent_predictions}, columns=['rent_final', 'city_final', 'rent_predictions'])
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
0        (0, 500]  40.909860      4  40.909860
1     (500, 1000]   4.822335    235   3.352132
2    (1000, 2000]   4.701700    505  -0.886717
3    (2000, 3000]   4.641735    113  -0.351083
4    (3000, 4000]   5.891771     50  -0.718262
5  (4000, 100000]   5.562821     63  -2.909349

###precision         Rent_bins     MdAPE  count      MdPE
0        (0, 500]  4.997207      2  4.358459
1     (500, 1000]  4.377220    210  1.721300
2    (1000, 2000]  5.280471    548 -0.458485
3    (2000, 3000]  4.611295    104 -0.369726
4    (3000, 4000]  5.388983     52  0.388802
5  (4000, 100000]  4.304390     54 -3.368959    
    
del(a,b,c,d)

a = train_eb.groupby('city_final').apply(median_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MdAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(median_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MdPE']
d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MdPE']], how='left', left_index=True, right_index=True)

#     city_final     MdAPE  count      MdPE
0       Austin  5.269674    386  0.219685
1  Chattanooga  3.533379     44  0.628023
2      Chicago  4.782825    287  0.049335
3     Columbia  4.633417     94 -1.034901
4       Dayton  4.813889     57  0.604861
5       Eugene  4.463514     44 -0.778226
6      Newyork  4.767884     58 -0.757873

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

        Rent_bins      MdAPE  count       MdPE
0        (0, 500]   9.255074      3   9.255074
1     (500, 1000]  12.761336    114  11.727714
2    (1000, 2000]  10.465910    201  -2.288328
3    (2000, 3000]  22.634432     52  -2.219706
4    (3000, 4000]  15.684898     17   3.806830
5  (4000, 100000]  23.650450     29 -23.650450
 
    
## precision        Rent_bins      MdAPE  count       MdPE
0        (0, 500]   6.125145      2  -1.904414
1     (500, 1000]  11.522535     93   0.462378
2    (1000, 2000]  13.001991    231   1.571965
3    (2000, 3000]  10.898169     43  -0.798716
4    (3000, 4000]  16.442479     21  14.046448
5  (4000, 100000]  13.907346     26   0.122617    
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
0       Austin  11.198596    166  -0.328065
1  Chattanooga  12.711016     19  -4.471127
2      Chicago  13.135026    123   3.179448
3     Columbia  10.465693     40   2.868435
4       Dayton  16.800068     24  13.202665
5       Eugene  14.482510     19  -1.841341
6      Newyork  16.612614     25 -14.943142

del(a,b,c,d)



