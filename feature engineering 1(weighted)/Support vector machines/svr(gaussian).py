

#Gaussian RBF Kernel
parameters = [{'kernel': ['rbf'],'gamma': [1e-4, 1e-3,0.001, 0.01, 0.1, 0.2, 0.5,1,3,5,10],
                'C':[1,5,10,25,50,75,100],'epsilon':[0.1,0.2,0.3,0.5]}]

print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(SVR(), parameters, cv=5)
clf.fit(ind_scaled, dep_scaled)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
sv2=SVR (kernel = 'rbf', C=1 ,gamma=0.01 ,epsilon = 0.3)
sv2.fit(ind_scaled, dep_scaled)

#Cross-validated Train errors
temp2 = sv2.predict(ind_scaled)
rent_predictions = (temp2*dep.std()) + dep.mean()
sv2_mse = mean_squared_error(dep, rent_predictions)
sv2_rmse = numpy.sqrt(sv2_mse)
sv2_rmse
#710.3729
sklearn.metrics.r2_score(dep, rent_predictions)*100
#65.4582

mean_absolute_percentage_error(dep, rent_predictions)
#25.37
mean_percentage_error(dep ,rent_predictions)
#-9.27



#Error buckets - by MAPE/MPE, rent range and city

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


                 len
                MAPE
MAPE_bins           
(0, 10]        273.0
(10, 20]       254.0
(20, 30]       165.0
(30, 50]       164.0
(50, 100]       91.0
(100, 100000]   23.0
#calculate proportion

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])
                  len
                   MPE
MPE_bins              
(-100000, -100]   23.0
(-100, -50]       71.0
(-50, -30]       106.0
(-30, -20]        99.0
(-20, -10]       115.0
(-10, 0]         143.0
(0, 10]          130.0
(10, 20]         139.0
(20, 30]          66.0
(30, 50]          58.0
(50, 100]         20.0
(100, 100000]      NaN

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

        Rent_bins        MAPE  count         MPE
0        (0, 500]  145.261076      4  145.261076
1     (500, 1000]   45.295186    235   45.285152
2    (1000, 2000]   16.228007    505    2.674523
3    (2000, 3000]   19.739373    113  -10.138620
4    (3000, 4000]   22.095372     50  -13.239017
5  (4000, 100000]   29.424353     63  -28.090550
    
    
    
del(a,b,c,d)

a = train_eb.groupby('city_final').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MPE']

d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)

    city_final       MAPE  count        MPE
0       Austin  19.982803    386   3.847066
1  Chattanooga  25.276049     44  19.739352
2      Chicago  24.129894    287   3.463358
3     Columbia  31.130441     94  23.756163
4       Dayton  55.046762     57  51.582573
5       Eugene  36.804624     44  25.738914
6      Newyork  20.267078     58 -11.326064



train_eb.to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/mape_test.csv')



joblib.dump(sv1, "sv1.pkl")
#my_model_loaded = joblib.load("sv1.pkl")





#test data

temp3 = sv2.predict(ind_test_scaled)
rent_predictions_test = temp3*dep.std() + dep.mean()
sv3_mse_test = mean_squared_error(dep_test, rent_predictions_test)
sv3_rmse_test = numpy.sqrt(sv3_mse_test)
sv3_rmse_test
#800.0078251271876
sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#62.36
mean_absolute_percentage_error(dep_test, rent_predictions_test)
#29.50
mean_percentage_error(dep_test, rent_predictions_test)
#-11.12




###test data
#Error buckets - by MAPE/MPE, rent range and city

a=strat_test_set.iloc[:,90].values
b=strat_test_set.iloc[:,91].values
test_eb = pandas.DataFrame({'rent_final': a, 'city_final': b, 'rent_predictions_test': rent_predictions_test}, columns=['rent_final', 'city_final', 'rent_predictions_test'])
del(a,b)
test_eb['MAPE'] = (numpy.abs(test_eb['rent_final'] - test_eb['rent_predictions_test']) / test_eb['rent_final']) * 100
test_eb['MPE'] = ((test_eb['rent_final'] - test_eb['rent_predictions_test']) / test_eb['rent_final']) * 100
bins = [0, 10, 20, 30, 50, 100, 100000]
test_eb['MAPE_bins'] = pandas.cut(test_eb['MAPE'], bins)
bins = [-100000, -100, -50, -30, -20, -10,  0, 10, 20, 30, 50, 100, 100000]
test_eb['MPE_bins'] = pandas.cut(test_eb['MPE'], bins)
pandas.pivot_table(test_eb,index=["MAPE_bins"],values=["MAPE"],aggfunc=[len])
                len
               MAPE
MAPE_bins          
(0, 10]        97.0
(10, 20]       90.0
(20, 30]       75.0
(30, 50]       91.0
(50, 100]      52.0
(100, 100000]  11.0
    
    
#calculate proportion

pandas.pivot_table(test_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])
                  len
                  MPE
MPE_bins             
(-100000, -100]  11.0
(-100, -50]      46.0
(-50, -30]       49.0
(-30, -20]       33.0
(-20, -10]       56.0
(-10, 0]         41.0
(0, 10]          56.0
(10, 20]         34.0
(20, 30]         42.0
(30, 50]         42.0
(50, 100]         6.0
(100, 100000]     NaN


bins = [0, 500, 1000, 2000, 3000, 4000, 100000]
test_eb['Rent_bins'] = pandas.cut(test_eb['rent_final'], bins)

def mean_absolute_percentage_error_df(df,y_true, y_pred): 
    return numpy.mean(numpy.abs(df[y_pred] - df[y_true]) / df[y_true]) * 100

def mean_percentage_error_df(df, y_true, y_pred): 
    return numpy.mean((df[y_pred] - df[y_true]) / df[y_true]) * 100

a = test_eb.groupby('Rent_bins').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions_test').reset_index()
a.columns = ['Rent_bins','MAPE']
b = test_eb.groupby('Rent_bins').size().reset_index()
b.columns = ['Rent_bins','count']
c = test_eb.groupby('Rent_bins').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions_test').reset_index()
c.columns = ['Rent_bins','MPE']

d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)
        Rent_bins        MAPE  count         MPE
0        (0, 500]  164.196479      3  164.196479
1     (500, 1000]   46.393256    114   46.391094
2    (1000, 2000]   17.785871    201    3.222204
3    (2000, 3000]   27.864235     52   -9.686148
4    (3000, 4000]   25.777682     17  -15.904822
5  (4000, 100000]   35.541741     29  -35.407622
    
del(a,b,c,d)

a = test_eb.groupby('city_final').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions_test').reset_index()
a.columns = ['city_final','MAPE']
b = test_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = test_eb.groupby('city_final').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions_test').reset_index()
c.columns = ['city_final','MPE']

d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)
    city_final       MAPE  count        MPE
0       Austin  21.004156    166   4.585509
1  Chattanooga  25.970154     19  23.049796
2      Chicago  29.541856    123   6.767152
3     Columbia  35.643196     40  31.477963
4       Dayton  73.733407     24  71.604028
5       Eugene  38.027131     19  12.681203
6      Newyork  29.705429     25 -24.884160







