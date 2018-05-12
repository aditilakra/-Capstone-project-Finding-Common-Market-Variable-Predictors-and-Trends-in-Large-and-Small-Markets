

#polynomial
parameters = [{'kernel': ['poly'],'degree':[2,3,4,5],
    'C':[1,5,10,25,50,75,100],'epsilon':[0.1,0.2,0.3,0.5]}]

print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(SVR(), parameters, cv=5)
clf.fit(ind_scaled, dep_scaled)
print(clf.best_params_)


svm_poly_reg = SVR(kernel="poly", degree=2, C=1, epsilon=0.3)
svm_poly_reg.fit(ind_scaled, dep_scaled)


#Cross-validated Train errors
temp_poly = svm_poly_reg.predict(ind_scaled)
rent_predictions = (temp_poly*dep.std()) + dep.mean()
svpoly_mse = mean_squared_error(dep, rent_predictions)
svpoly_rmse = numpy.sqrt(svpoly_mse)
svpoly_rmse
#659.8
sklearn.metrics.r2_score(dep, rent_predictions)*100
#70.2
mean_absolute_percentage_error(dep, rent_predictions )
#27.34
mean_percentage_error(dep, rent_predictions)
#-9.22




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
#                 len
#                MAPE
#MAPE_bins           
#(0, 10]        248.0
#(10, 20]       245.0
#(20, 30]       180.0
#(30, 50]       163.0
#(50, 100]      104.0
#(100, 100000]   30.0
#calculate proportion

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])
#                   len
#                   MPE
#MPE_bins              
#(-100000, -100]   30.0
#(-100, -50]       80.0
#(-50, -30]        99.0
#(-30, -20]        92.0
#(-20, -10]       110.0
#(-10, 0]         108.0
#(0, 10]          140.0
#(10, 20]         135.0
#(20, 30]          88.0
#(30, 50]          64.0
#(50, 100]         24.0
#(100, 100000]      NaN

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
0        (0, 500]  143.401364      4  143.401364
1     (500, 1000]   55.158844    235   55.158844
2    (1000, 2000]   15.336485    505   -0.512934
3    (2000, 3000]   22.320382    113  -18.015067
4    (3000, 4000]   25.376033     50  -20.938444
5  (4000, 100000]   23.004552     63  -19.805995
    
    
    
    
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
0       Austin  19.630463    386   1.903023
1  Chattanooga  36.198176     44  33.489591
2      Chicago  23.774936    287  -5.731539
3     Columbia  42.006528     94  38.541756
4       Dayton  70.407367     57  68.101466
5       Eugene  38.033890     44  31.248222
6      Newyork  15.385188     58  -8.556691

train_eb.to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/mape_test.csv')



joblib.dump(sv1, "sv1.pkl")
#my_model_loaded = joblib.load("sv1.pkl")





##test data 
temp_poly = svm_poly_reg.predict(ind_test_scaled)
rent_predictions_test = temp_poly*dep.std() + dep.mean()
svm_poly_reg_test = mean_squared_error(dep_test, rent_predictions_test)
svm_poly_reg_test = numpy.sqrt(svm_poly_reg_test)
svm_poly_reg_test
#907.313
sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#51.58
mean_absolute_percentage_error(dep_test, rent_predictions_test)
#35.2
mean_percentage_error(dep_test, rent_predictions_test)
#-12.177



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
(0, 10]        84.0
(10, 20]       74.0
(20, 30]       81.0
(30, 50]       94.0
(50, 100]      59.0
(100, 100000]  24.0
    

#calculate proportion

pandas.pivot_table(test_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])
                  len
                  MPE
MPE_bins             
(-100000, -100]  24.0
(-100, -50]      44.0
(-50, -30]       42.0
(-30, -20]       41.0
(-20, -10]       34.0
(-10, 0]         40.0
(0, 10]          44.0
(10, 20]         40.0
(20, 30]         40.0
(30, 50]         52.0
(50, 100]        15.0
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
0        (0, 500]  245.750711      3  245.750711
1     (500, 1000]   59.370929    114   59.370929
2    (1000, 2000]   18.479279    201   -0.423346
3    (2000, 3000]   32.254453     52  -18.277299
4    (3000, 4000]   37.217910     17  -27.745485
5  (4000, 100000]   38.404054     29  -32.156923
    
    
del(a,b,c,d)

a = test_eb.groupby('city_final').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions_test').reset_index()
a.columns = ['city_final','MAPE']
b = test_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = test_eb.groupby('city_final').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions_test').reset_index()
c.columns = ['city_final','MPE']

d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)


    city_final        MAPE  count         MPE
0       Austin   23.918427    166    3.726372
1  Chattanooga   32.633267     19   31.152738
2      Chicago   32.257100    123   -2.206075
3     Columbia   51.939181     40   49.916841
4       Dayton  105.308327     24  104.569981
5       Eugene   31.344187     19   11.639634
6      Newyork   35.396228     25  -24.033476

