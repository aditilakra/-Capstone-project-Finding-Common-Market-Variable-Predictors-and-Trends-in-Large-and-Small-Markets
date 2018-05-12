from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.svm import SVR

#Support Vector Regression

sc = StandardScaler(with_mean=True, with_std=True)
ind_scaled = sc.fit_transform(ind)
dep_scaled = (dep - dep.mean())/dep.std()
ind_scaled=pandas.DataFrame(ind_scaled)
ind_scaled.columns = [('weighted_rent_final', '0'),  ('weighted_rent_final', '1'),
       ('weighted_rent_final', '10'), ('weighted_rent_final', '11'),
       ('weighted_rent_final', '12'), ('weighted_rent_final', '13'),
       ('weighted_rent_final', '14'), ('weighted_rent_final', '15'),
       ('weighted_rent_final', '16'), ('weighted_rent_final', '17'),
       ('weighted_rent_final', '18'), ('weighted_rent_final', '19'),
        ('weighted_rent_final', '2'), ('weighted_rent_final', '20'),
       ('weighted_rent_final', '21'), ('weighted_rent_final', '22'),
       ('weighted_rent_final', '23'), ('weighted_rent_final', '24'),
       ('weighted_rent_final', '25'), ('weighted_rent_final', '26'),
       ('weighted_rent_final', '27'), ('weighted_rent_final', '28'),
       ('weighted_rent_final', '29'),  ('weighted_rent_final', '3'),
        ('weighted_rent_final', '4'),  ('weighted_rent_final', '5'),
        ('weighted_rent_final', '6'),  ('weighted_rent_final', '7'),
        ('weighted_rent_final', '8'),  ('weighted_rent_final', '9')]

parameters = {'C':[1,5,10,25,50,75,100],'epsilon':[0.1,0.2,0.3,0.5]}
#parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.3,0.5]}
svr_linear = LinearSVR()
svr_gridsearch = GridSearchCV(svr_linear, parameters,cv=5,scoring='neg_mean_squared_error')
svr_gridsearch.fit(ind_scaled,dep_scaled)
svr_gridsearch.best_params_
svr_gridsearch.best_estimator_

sv1=LinearSVR(C=1, dual=True, epsilon=0.2, fit_intercept=True,
     intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
     random_state=None, tol=0.0001, verbose=0)
sv1.fit(ind_scaled, dep_scaled)

#Cross-validated Train errors
temp = sv1.predict(ind_scaled)
rent_predictions = temp*dep.std() + dep.mean()
sv1_mse = mean_squared_error(dep, rent_predictions)
sv1_rmse = numpy.sqrt(sv1_mse)
sv1_rmse
#797.52
sklearn.metrics.r2_score(dep, rent_predictions)*100
#56.46
def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100

def mean_percentage_error(y_true, y_pred): 
    return numpy.mean((y_true - y_pred) / y_true) * 100

mean_absolute_percentage_error(dep, rent_predictions)
#26.12
mean_percentage_error(dep, rent_predictions)
#-6.85



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
(0, 10]        266.0
(10, 20]       245.0
(20, 30]       163.0
(30, 50]       176.0
(50, 100]       99.0
(100, 100000]   21.0

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])

                   len
                   MPE
MPE_bins              
(-100000, -100]   21.0
(-100, -50]       72.0
(-50, -30]        96.0
(-30, -20]        82.0
(-20, -10]       119.0
(-10, 0]         140.0
(0, 10]          126.0
(10, 20]         126.0
(20, 30]          81.0
(30, 50]          80.0
(50, 100]         27.0
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
0        (0, 500]  159.309300      4  159.309300
1     (500, 1000]   38.656531    235   37.488132
2    (1000, 2000]   19.170013    505    2.563939
3    (2000, 3000]   21.308599    113  -11.703408
4    (3000, 4000]   25.815459     50  -13.557110
5  (4000, 100000]   35.602331     63  -33.186035

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
0       Austin  21.748275    386   3.423051
1  Chattanooga  23.295607     44  11.256908
2      Chicago  26.449835    287   3.874078
3     Columbia  25.944898     94  14.064591
4       Dayton  48.849321     57  41.791305
5       Eugene  33.825739     44  16.301050
6      Newyork  27.956243     58 -12.055311

train_eb.to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/mape_test.csv')
del(a,b,c,d)
joblib.dump(sv1, "sv1.pkl")
#my_model_loaded = joblib.load("sv1.pkl")




#test error
ind_test = strat_test_set[[('weighted_rent_final', '0'),  ('weighted_rent_final', '1'),
       ('weighted_rent_final', '10'), ('weighted_rent_final', '11'),
       ('weighted_rent_final', '12'), ('weighted_rent_final', '13'),
       ('weighted_rent_final', '14'), ('weighted_rent_final', '15'),
       ('weighted_rent_final', '16'), ('weighted_rent_final', '17'),
       ('weighted_rent_final', '18'), ('weighted_rent_final', '19'),
        ('weighted_rent_final', '2'), ('weighted_rent_final', '20'),
       ('weighted_rent_final', '21'), ('weighted_rent_final', '22'),
       ('weighted_rent_final', '23'), ('weighted_rent_final', '24'),
       ('weighted_rent_final', '25'), ('weighted_rent_final', '26'),
       ('weighted_rent_final', '27'), ('weighted_rent_final', '28'),
       ('weighted_rent_final', '29'),  ('weighted_rent_final', '3'),
        ('weighted_rent_final', '4'),  ('weighted_rent_final', '5'),
        ('weighted_rent_final', '6'),  ('weighted_rent_final', '7'),
        ('weighted_rent_final', '8'),  ('weighted_rent_final', '9')]].copy()
dep_test = strat_test_set['rent_final'].copy()

ind_test_scaled = sc.transform(ind_test) 
temp1 = sv1.predict(ind_test_scaled)
rent_predictions_test = temp1*dep.std() + dep.mean()
sv1_mse_test = mean_squared_error(dep_test, rent_predictions_test)
sv1_rmse_test = numpy.sqrt(sv1_mse_test)
sv1_rmse_test
#809.86
sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#61.42
mean_absolute_percentage_error(dep_test, rent_predictions_test)
#29.00
mean_percentage_error(dep_test, rent_predictions_test)
#-8.61

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
(0, 10]        96.0
(10, 20]       82.0
(20, 30]       85.0
(30, 50]       95.0
(50, 100]      49.0
(100, 100000]   9.0
    

#calculate proportion

pandas.pivot_table(test_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])
                  len
                  MPE
MPE_bins             
(-100000, -100]   9.0
(-100, -50]      40.0
(-50, -30]       45.0
(-30, -20]       44.0
(-20, -10]       43.0
(-10, 0]         45.0
(0, 10]          51.0
(10, 20]         39.0
(20, 30]         41.0
(30, 50]         50.0
(50, 100]         9.0
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
0        (0, 500]  156.692127      3  156.692127
1     (500, 1000]   39.060966    114   38.297510
2    (1000, 2000]   21.076447    201    3.244750
3    (2000, 3000]   26.706986     52  -11.007279
4    (3000, 4000]   26.909179     17  -15.818439
5  (4000, 100000]   36.588477     29  -36.588477
    
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
0       Austin  22.449601    166   3.552941
1  Chattanooga  24.914780     19  17.950446
2      Chicago  30.327057    123   7.687030
3     Columbia  27.890335     40  18.727745
4       Dayton  63.796268     24  58.879310
5       Eugene  38.073879     19   8.657847

6      Newyork  30.652295     25 -24.689371
