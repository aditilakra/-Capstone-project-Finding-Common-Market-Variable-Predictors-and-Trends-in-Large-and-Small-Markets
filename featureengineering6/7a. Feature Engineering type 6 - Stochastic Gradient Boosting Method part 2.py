from sklearn.ensemble import GradientBoostingRegressor
import csv
import pandas
import math
from sklearn.model_selection import GridSearchCV
import numpy

#GBM for rental predictions on rents as a ratio of zipcode median gross rent (dependent variable transformed)

strat_train_set_wo_MGR1 = pandas.read_csv('/Users/aditilakra/Desktop/capstone/featureengineering6/strat_train_set_wo_MGR1.csv')
strat_test_set_wo_MGR1 = pandas.read_csv('/Users/aditilakra/Desktop/capstone/featureengineering6/strat_test_set_wo_MGR1.csv')
strat_train_set_wo_MGR1.drop('Unnamed: 0', axis=1, inplace=True)
strat_test_set_wo_MGR1.drop('Unnamed: 0', axis=1, inplace=True)

param_grid = [
{'learning_rate': [0.01,0.03,0.05], 'n_estimators': [200,300,400], 'max_depth': [3,4], 'subsample': [0.33,0.67,1], 'max_features':["sqrt","log2"], 'loss':["ls",  "huber"] },
]

gbrt = GradientBoostingRegressor()
grid_search = GridSearchCV(gbrt, param_grid, cv=5)
strat_train_set_wo_MGR1['ratio1'] = strat_train_set_wo_MGR1["('rent_final', '0')"]/strat_train_set_wo_MGR1["('SE_T104_001', '0')"]
strat_train_set_wo_MGR1['ratio2'] = strat_train_set_wo_MGR1["('rent_final', '1')"]/strat_train_set_wo_MGR1["('SE_T104_001', '1')"]
strat_train_set_wo_MGR1['ratio3'] = strat_train_set_wo_MGR1["('rent_final', '2')"]/strat_train_set_wo_MGR1["('SE_T104_001', '2')"]
strat_train_set_wo_MGR1['ratio4'] = strat_train_set_wo_MGR1["('rent_final', '3')"]/strat_train_set_wo_MGR1["('SE_T104_001', '3')"]
strat_train_set_wo_MGR1['ratio5'] = strat_train_set_wo_MGR1["('rent_final', '4')"]/strat_train_set_wo_MGR1["('SE_T104_001', '4')"]
strat_train_set_wo_MGR1['ratio6'] = strat_train_set_wo_MGR1["('rent_final', '5')"]/strat_train_set_wo_MGR1["('SE_T104_001', '5')"]
strat_train_set_wo_MGR1['ratio7'] = strat_train_set_wo_MGR1["('rent_final', '6')"]/strat_train_set_wo_MGR1["('SE_T104_001', '6')"]
strat_train_set_wo_MGR1['ratio8'] = strat_train_set_wo_MGR1["('rent_final', '7')"]/strat_train_set_wo_MGR1["('SE_T104_001', '7')"]
strat_train_set_wo_MGR1['ratio9'] = strat_train_set_wo_MGR1["('rent_final', '8')"]/strat_train_set_wo_MGR1["('SE_T104_001', '8')"]
strat_train_set_wo_MGR1['ratio10'] = strat_train_set_wo_MGR1["('rent_final', '9')"]/strat_train_set_wo_MGR1["('SE_T104_001', '9')"]
strat_train_set_wo_MGR1['ratio11'] = strat_train_set_wo_MGR1["('rent_final', '10')"]/strat_train_set_wo_MGR1["('SE_T104_001', '10')"]
strat_train_set_wo_MGR1['ratio12'] = strat_train_set_wo_MGR1["('rent_final', '11')"]/strat_train_set_wo_MGR1["('SE_T104_001', '11')"]
strat_train_set_wo_MGR1['ratio13'] = strat_train_set_wo_MGR1["('rent_final', '12')"]/strat_train_set_wo_MGR1["('SE_T104_001', '12')"]
strat_train_set_wo_MGR1['ratio14'] = strat_train_set_wo_MGR1["('rent_final', '13')"]/strat_train_set_wo_MGR1["('SE_T104_001', '13')"]
strat_train_set_wo_MGR1['ratio15'] = strat_train_set_wo_MGR1["('rent_final', '14')"]/strat_train_set_wo_MGR1["('SE_T104_001', '14')"]
strat_train_set_wo_MGR1['ratio16'] = strat_train_set_wo_MGR1["('rent_final', '15')"]/strat_train_set_wo_MGR1["('SE_T104_001', '15')"]
strat_train_set_wo_MGR1['ratio17'] = strat_train_set_wo_MGR1["('rent_final', '16')"]/strat_train_set_wo_MGR1["('SE_T104_001', '16')"]
strat_train_set_wo_MGR1['ratio18'] = strat_train_set_wo_MGR1["('rent_final', '17')"]/strat_train_set_wo_MGR1["('SE_T104_001', '17')"]
strat_train_set_wo_MGR1['ratio19'] = strat_train_set_wo_MGR1["('rent_final', '18')"]/strat_train_set_wo_MGR1["('SE_T104_001', '18')"]
strat_train_set_wo_MGR1['ratio20'] = strat_train_set_wo_MGR1["('rent_final', '19')"]/strat_train_set_wo_MGR1["('SE_T104_001', '19')"]
strat_train_set_wo_MGR1['ratio21'] = strat_train_set_wo_MGR1["('rent_final', '20')"]/strat_train_set_wo_MGR1["('SE_T104_001', '20')"]
strat_train_set_wo_MGR1['ratio22'] = strat_train_set_wo_MGR1["('rent_final', '21')"]/strat_train_set_wo_MGR1["('SE_T104_001', '21')"]
strat_train_set_wo_MGR1['ratio23'] = strat_train_set_wo_MGR1["('rent_final', '22')"]/strat_train_set_wo_MGR1["('SE_T104_001', '22')"]
strat_train_set_wo_MGR1['ratio24'] = strat_train_set_wo_MGR1["('rent_final', '23')"]/strat_train_set_wo_MGR1["('SE_T104_001', '23')"]
strat_train_set_wo_MGR1['ratio25'] = strat_train_set_wo_MGR1["('rent_final', '24')"]/strat_train_set_wo_MGR1["('SE_T104_001', '24')"]
strat_train_set_wo_MGR1['ratio26'] = strat_train_set_wo_MGR1["('rent_final', '25')"]/strat_train_set_wo_MGR1["('SE_T104_001', '25')"]
strat_train_set_wo_MGR1['ratio27'] = strat_train_set_wo_MGR1["('rent_final', '26')"]/strat_train_set_wo_MGR1["('SE_T104_001', '26')"]
strat_train_set_wo_MGR1['ratio28'] = strat_train_set_wo_MGR1["('rent_final', '27')"]/strat_train_set_wo_MGR1["('SE_T104_001', '27')"]
strat_train_set_wo_MGR1['ratio29'] = strat_train_set_wo_MGR1["('rent_final', '28')"]/strat_train_set_wo_MGR1["('SE_T104_001', '28')"]
strat_train_set_wo_MGR1['ratio30'] = strat_train_set_wo_MGR1["('rent_final', '29')"]/strat_train_set_wo_MGR1["('SE_T104_001', '29')"]

ind = strat_train_set_wo_MGR1[["ratio1","ratio2","ratio3","ratio4","ratio5","ratio6","ratio7","ratio8",
                               "ratio9","ratio10","ratio11","ratio12","ratio13","ratio14","ratio15","ratio16",
                               "ratio17","ratio18","ratio19","ratio20","ratio21","ratio22","ratio23","ratio24",
                               "ratio25","ratio26","ratio27","ratio28","ratio29","ratio30"]]

log=lambda x: math.log10(x+1)
ind=ind.applymap(log)

dep =(strat_train_set_wo_MGR1['rent_final']/strat_train_set_wo_MGR1['SE_T104_001']).copy()
dep=numpy.log10(dep+1)

grid_search.fit(ind, dep)
grid_search.best_params_
grid_search.best_estimator_

gbm8 =GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.01, loss='huber', max_depth=3,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=400,
             presort='auto', random_state=None, subsample=0.33, verbose=0,
             warm_start=False)
gbm8.fit(ind, dep)

#Cross-validated Train errors
temp = gbm8.predict(ind)
temp1 = numpy.power(10, temp) - 1
rent_predictions = temp1*strat_train_set_wo_MGR1['SE_T104_001']
dep = strat_train_set_wo_MGR1['rent_final']
rf1_mse = mean_squared_error(dep, rent_predictions)
rf1_rmse = numpy.sqrt(rf1_mse)
rf1_rmse
#450.63
sklearn.metrics.r2_score(dep, rent_predictions)*100
#86.09
def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.sum(numpy.abs(y_pred - y_true))/numpy.sum(y_true)*100

def mean_percentage_error(y_true, y_pred): 
    return numpy.sum(y_pred - y_true) / numpy.sum(y_true) * 100

mean_absolute_percentage_error(dep, rent_predictions)
#17.24%
mean_percentage_error(dep, rent_predictions)
#0.77%

#test error
strat_test_set_wo_MGR1['ratio1'] = strat_test_set_wo_MGR1["('rent_final', '0')"]/strat_test_set_wo_MGR1["('SE_T104_001', '0')"]
strat_test_set_wo_MGR1['ratio2'] = strat_test_set_wo_MGR1["('rent_final', '1')"]/strat_test_set_wo_MGR1["('SE_T104_001', '1')"]
strat_test_set_wo_MGR1['ratio3'] = strat_test_set_wo_MGR1["('rent_final', '2')"]/strat_test_set_wo_MGR1["('SE_T104_001', '2')"]
strat_test_set_wo_MGR1['ratio4'] = strat_test_set_wo_MGR1["('rent_final', '3')"]/strat_test_set_wo_MGR1["('SE_T104_001', '3')"]
strat_test_set_wo_MGR1['ratio5'] = strat_test_set_wo_MGR1["('rent_final', '4')"]/strat_test_set_wo_MGR1["('SE_T104_001', '4')"]
strat_test_set_wo_MGR1['ratio6'] = strat_test_set_wo_MGR1["('rent_final', '5')"]/strat_test_set_wo_MGR1["('SE_T104_001', '5')"]
strat_test_set_wo_MGR1['ratio7'] = strat_test_set_wo_MGR1["('rent_final', '6')"]/strat_test_set_wo_MGR1["('SE_T104_001', '6')"]
strat_test_set_wo_MGR1['ratio8'] = strat_test_set_wo_MGR1["('rent_final', '7')"]/strat_test_set_wo_MGR1["('SE_T104_001', '7')"]
strat_test_set_wo_MGR1['ratio9'] = strat_test_set_wo_MGR1["('rent_final', '8')"]/strat_test_set_wo_MGR1["('SE_T104_001', '8')"]
strat_test_set_wo_MGR1['ratio10'] = strat_test_set_wo_MGR1["('rent_final', '9')"]/strat_test_set_wo_MGR1["('SE_T104_001', '9')"]
strat_test_set_wo_MGR1['ratio11'] = strat_test_set_wo_MGR1["('rent_final', '10')"]/strat_test_set_wo_MGR1["('SE_T104_001', '10')"]
strat_test_set_wo_MGR1['ratio12'] = strat_test_set_wo_MGR1["('rent_final', '11')"]/strat_test_set_wo_MGR1["('SE_T104_001', '11')"]
strat_test_set_wo_MGR1['ratio13'] = strat_test_set_wo_MGR1["('rent_final', '12')"]/strat_test_set_wo_MGR1["('SE_T104_001', '12')"]
strat_test_set_wo_MGR1['ratio14'] = strat_test_set_wo_MGR1["('rent_final', '13')"]/strat_test_set_wo_MGR1["('SE_T104_001', '13')"]
strat_test_set_wo_MGR1['ratio15'] = strat_test_set_wo_MGR1["('rent_final', '14')"]/strat_test_set_wo_MGR1["('SE_T104_001', '14')"]
strat_test_set_wo_MGR1['ratio16'] = strat_test_set_wo_MGR1["('rent_final', '15')"]/strat_test_set_wo_MGR1["('SE_T104_001', '15')"]
strat_test_set_wo_MGR1['ratio17'] = strat_test_set_wo_MGR1["('rent_final', '16')"]/strat_test_set_wo_MGR1["('SE_T104_001', '16')"]
strat_test_set_wo_MGR1['ratio18'] = strat_test_set_wo_MGR1["('rent_final', '17')"]/strat_test_set_wo_MGR1["('SE_T104_001', '17')"]
strat_test_set_wo_MGR1['ratio19'] = strat_test_set_wo_MGR1["('rent_final', '18')"]/strat_test_set_wo_MGR1["('SE_T104_001', '18')"]
strat_test_set_wo_MGR1['ratio20'] = strat_test_set_wo_MGR1["('rent_final', '19')"]/strat_test_set_wo_MGR1["('SE_T104_001', '19')"]
strat_test_set_wo_MGR1['ratio21'] = strat_test_set_wo_MGR1["('rent_final', '20')"]/strat_test_set_wo_MGR1["('SE_T104_001', '20')"]
strat_test_set_wo_MGR1['ratio22'] = strat_test_set_wo_MGR1["('rent_final', '21')"]/strat_test_set_wo_MGR1["('SE_T104_001', '21')"]
strat_test_set_wo_MGR1['ratio23'] = strat_test_set_wo_MGR1["('rent_final', '22')"]/strat_test_set_wo_MGR1["('SE_T104_001', '22')"]
strat_test_set_wo_MGR1['ratio24'] = strat_test_set_wo_MGR1["('rent_final', '23')"]/strat_test_set_wo_MGR1["('SE_T104_001', '23')"]
strat_test_set_wo_MGR1['ratio25'] = strat_test_set_wo_MGR1["('rent_final', '24')"]/strat_test_set_wo_MGR1["('SE_T104_001', '24')"]
strat_test_set_wo_MGR1['ratio26'] = strat_test_set_wo_MGR1["('rent_final', '25')"]/strat_test_set_wo_MGR1["('SE_T104_001', '25')"]
strat_test_set_wo_MGR1['ratio27'] = strat_test_set_wo_MGR1["('rent_final', '26')"]/strat_test_set_wo_MGR1["('SE_T104_001', '26')"]
strat_test_set_wo_MGR1['ratio28'] = strat_test_set_wo_MGR1["('rent_final', '27')"]/strat_test_set_wo_MGR1["('SE_T104_001', '27')"]
strat_test_set_wo_MGR1['ratio29'] = strat_test_set_wo_MGR1["('rent_final', '28')"]/strat_test_set_wo_MGR1["('SE_T104_001', '28')"]
strat_test_set_wo_MGR1['ratio30'] = strat_test_set_wo_MGR1["('rent_final', '29')"]/strat_test_set_wo_MGR1["('SE_T104_001', '29')"]

ind_test = strat_test_set_wo_MGR1[["ratio1","ratio2","ratio3","ratio4","ratio5","ratio6","ratio7","ratio8",
                               "ratio9","ratio10","ratio11","ratio12","ratio13","ratio14","ratio15","ratio16",
                               "ratio17","ratio18","ratio19","ratio20","ratio21","ratio22","ratio23","ratio24",
                               "ratio25","ratio26","ratio27","ratio28","ratio29","ratio30"]]

ind_test=ind_test.applymap(log)

temp_test = gbm8.predict(ind_test)
temp_test1 = numpy.power(10, temp_test) - 1
rent_predictions_test = temp_test1*strat_test_set_wo_MGR1['SE_T104_001']
dep_test = strat_test_set_wo_MGR1['rent_final']

rf1_mse_test = mean_squared_error(dep_test, rent_predictions_test)
rf1_rmse_test = numpy.sqrt(rf1_mse_test)
rf1_rmse_test
# 748.66
sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#67.03
mean_absolute_percentage_error(dep_test, rent_predictions_test)
#24.78
mean_percentage_error(dep_test, rent_predictions_test)
#3.03

#Variable importance
feature_importances = gbm8.feature_importances_
feature_importances
attributes = ind.columns
pandas.DataFrame(sorted(zip(feature_importances, attributes), reverse=True)).to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/stoachastic_gbm_varimp_ratioofzipcoderent1.csv')

#Error buckets - by MAPE/MPE, rent range and city - train

a=strat_train_set_wo_MGR1.iloc[:,120].values
b=strat_train_set_wo_MGR1.iloc[:,121].values
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

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])

#

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

del(a,b,c,d)

#Error buckets - by MAPE/MPE, rent range and city - test

a=strat_test_set_wo_MGR1.iloc[:,120].values
b=strat_test_set_wo_MGR1.iloc[:,121].values
train_eb = pandas.DataFrame({'rent_final': a, 'city_final': b, 'rent_predictions': rent_predictions_test}, columns=['rent_final', 'city_final', 'rent_predictions'])
del(a,b)
train_eb['MAPE'] = (numpy.abs(train_eb['rent_final'] - train_eb['rent_predictions']) / train_eb['rent_final']) * 100
train_eb['MPE'] = ((train_eb['rent_final'] - train_eb['rent_predictions']) / train_eb['rent_final']) * 100
bins = [0, 10, 20, 30, 50, 100, 100000]
train_eb['MAPE_bins'] = pandas.cut(train_eb['MAPE'], bins)
bins = [-100000, -100, -50, -30, -20, -10,  0, 10, 20, 30, 50, 100, 100000]
train_eb['MPE_bins'] = pandas.cut(train_eb['MPE'], bins)
pandas.pivot_table(train_eb,index=["MAPE_bins"],values=["MAPE"],aggfunc=[len])

#

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])

#

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
#13.78%

median_percentage_error(dep, rent_predictions)
#4.47%

#test error

median_absolute_percentage_error(dep_test, rent_predictions_test)
#18.81%

median_percentage_error(dep_test, rent_predictions_test)
#10.42%

#Error buckets - by MAPE/MPE, rent range and city - train
a=strat_train_set_wo_MGR1.iloc[:,120].values
b=strat_train_set_wo_MGR1.iloc[:,121].values
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

del(a,b,c,d)

a = train_eb.groupby('city_final').apply(median_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MdAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(median_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MdPE']
d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MdPE']], how='left', left_index=True, right_index=True)

#    

del(a,b,c,d)

#Error buckets - by MAPE/MPE, rent range and city - test

a=strat_test_set_wo_MGR1.iloc[:,120].values
b=strat_test_set_wo_MGR1.iloc[:,121].values
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

#

del(a,b,c,d)

a = train_eb.groupby('city_final').apply(median_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MdAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(median_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MdPE']
d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MdPE']], how='left', left_index=True, right_index=True)

#

del(a,b,c,d)


