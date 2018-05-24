from sklearn.ensemble import GradientBoostingRegressor

#GBM for rental predictions on weighted rents

#n_features = 30
#numpy.sqrt(n_features)
#numpy.log2(n_features)
#param_grid = [
#{'learning_rate': [0.005,0.01,0.05,0.1,0.15], 'n_estimators': [300,500,700,900], 'max_depth': [2,4,6], 'subsample': [0.25,0.5,0.75,1], 'max_features':["auto","sqrt","log2"] },
#]

#param_grid = [
#{'learning_rate': [0.01,0.03,0.05,0.07,0.09], 'n_estimators': [300,500,700,900], 'max_depth': [2,3,4], 'subsample': [0.33,0.67,1], 'max_features':["auto","sqrt","log2"] },
#]

param_grid = [
{'learning_rate': [0.01,0.03,0.05,0.07,0.09], 'n_estimators': [200,300,400,500], 'max_depth': [2,3,4], 'subsample': [0.33,0.67,1], 'max_features':["auto","sqrt","log2"], 'loss':["ls", "lad", "huber"] },
]

gbrt = GradientBoostingRegressor()
grid_search = GridSearchCV(gbrt, param_grid, cv=5)

ind = strat_train_set[["('rent_final', '0')",  "('rent_final', '1')",
       "('rent_final', '10')", "('rent_final', '11')",
       "('rent_final', '12')", "('rent_final', '13')",
       "('rent_final', '14')", "('rent_final', '15')",
       "('rent_final', '16')", "('rent_final', '17')",
       "('rent_final', '18')", "('rent_final', '19')",
        "('rent_final', '2')", "('rent_final', '20')",
       "('rent_final', '21')", "('rent_final', '22')",
       "('rent_final', '23')", "('rent_final', '24')",
       "('rent_final', '25')", "('rent_final', '26')",
      "('rent_final', '27')", "('rent_final', '28')",
       "('rent_final', '29')",  "('rent_final', '3')",
        "('rent_final', '4')",  "('rent_final', '5')",
        "('rent_final', '6')",  "('rent_final', '7')",
        "('rent_final', '8')",  "('rent_final', '9')"]].copy()
dep = strat_train_set['rent_final'].copy()
grid_search.fit(ind, dep)
grid_search.best_params_
grid_search.best_estimator_

#cvres = grid_search.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#    print(numpy.sqrt(-mean_score), params)
#gbm4 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#             learning_rate=0.01, loss='ls', max_depth=4,
#             max_features='sqrt', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=300,
#             presort='auto', random_state=None, subsample=0.33, verbose=0,
#             warm_start=False)
#gbm4.fit(ind, dep)

gbm5 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.01, loss='ls', max_depth=4,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=500,
             presort='auto', random_state=None, subsample=0.33, verbose=0,
             warm_start=False)
gbm5.fit(ind, dep)

#Cross-validated Train errors
rent_predictions = gbm5.predict(ind)
rf1_mse = mean_squared_error(dep, rent_predictions)
rf1_rmse = numpy.sqrt(rf1_mse)
rf1_rmse
#402.57
sklearn.metrics.r2_score(dep, rent_predictions)*100
#88.9
def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.mean(numpy.abs(y_pred - y_true) / y_true) * 100

def mean_percentage_error(y_true, y_pred): 
    return numpy.mean((y_pred - y_true) / y_true) * 100

mean_absolute_percentage_error(dep, rent_predictions)
#18.4%
mean_percentage_error(dep, rent_predictions)
#7.12%

#test error
ind_test = strat_test_set[["('rent_final', '0')",  "('rent_final', '1')",
       "('rent_final', '10')", "('rent_final', '11')",
       "('rent_final', '12')", "('rent_final', '13')",
       "('rent_final', '14')", "('rent_final', '15')",
       "('rent_final', '16')", "('rent_final', '17')",
       "('rent_final', '18')", "('rent_final', '19')",
        "('rent_final', '2')", "('rent_final', '20')",
       "('rent_final', '21')", "('rent_final', '22')",
       "('rent_final', '23')", "('rent_final', '24')",
       "('rent_final', '25')", "('rent_final', '26')",
      "('rent_final', '27')", "('rent_final', '28')",
       "('rent_final', '29')",  "('rent_final', '3')",
        "('rent_final', '4')",  "('rent_final', '5')",
        "('rent_final', '6')",  "('rent_final', '7')",
        "('rent_final', '8')",  "('rent_final', '9')"]].copy()
dep_test = strat_test_set['rent_final'].copy()

rent_predictions_test = gbm5.predict(ind_test)
rf1_mse_test = mean_squared_error(dep_test, rent_predictions_test)
rf1_rmse_test = numpy.sqrt(rf1_mse_test)
rf1_rmse_test
#717.02
sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#69.76
mean_absolute_percentage_error(dep_test, rent_predictions_test)
#23.68
mean_percentage_error(dep_test, rent_predictions_test)
#9.68

#Variable importance
feature_importances = gbm5.feature_importances_
feature_importances
attributes = ind.columns
pandas.DataFrame(sorted(zip(feature_importances, attributes), reverse=True)).to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/stoachastic_gbm_varimp_origrent.csv')

#Error buckets - by MAPE/MPE, rent range and city - train

a=strat_train_set.iloc[:,90].values
b=strat_train_set.iloc[:,91].values
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
    return numpy.mean(numpy.abs(df[y_pred] - df[y_true]) / df[y_true]) * 100

def mean_percentage_error_df(df, y_true, y_pred): 
    return numpy.mean((df[y_pred] - df[y_true]) / df[y_true]) * 100

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

a=strat_test_set.iloc[:,90].values
b=strat_test_set.iloc[:,91].values
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
    return numpy.mean(numpy.abs(df[y_pred] - df[y_true]) / df[y_true]) * 100

def mean_percentage_error_df(df, y_true, y_pred): 
    return numpy.mean((df[y_pred] - df[y_true]) / df[y_true]) * 100

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

#joblib.dump(gbm4, "gbm4.pkl")
#gbm4 = joblib.load("gbm4.pkl")

joblib.dump(gbm5, "gbm5.pkl")
#gbm5 = joblib.load("gbm5.pkl")
