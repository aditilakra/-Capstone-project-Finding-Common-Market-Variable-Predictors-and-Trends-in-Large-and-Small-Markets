#Random Forests for variable importance & GBM for rental predictions on weighted rents

n_features = 30
numpy.sqrt(n_features)
numpy.log2(n_features)
param_grid = [
{'n_estimators': [100, 200, 300, 400, 500], 'max_features': [5,10,15], 'max_depth': [1,2,3,4,5,6]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error')
ind = strat_train_set[[('weighted_rent_final', '0'),  ('weighted_rent_final', '1'),
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
dep = strat_train_set['rent_final'].copy()
grid_search.fit(ind, dep)
grid_search.best_params_
grid_search.best_estimator_
#cvres = grid_search.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#    print(numpy.sqrt(-mean_score), params)
rf1 = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=6,
           max_features=5, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
rf1.fit(ind, dep)

#Cross-validated Train errors
rent_predictions = rf1.predict(ind)
rf1_mse = mean_squared_error(dep, rent_predictions)
rf1_rmse = numpy.sqrt(rf1_mse)
rf1_rmse
#495.51
sklearn.metrics.r2_score(dep, rent_predictions)*100
#83.19
def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100

def mean_percentage_error(y_true, y_pred): 
    return numpy.mean((y_true - y_pred) / y_true) * 100

mean_absolute_percentage_error(dep, rent_predictions)
#23.7729
mean_percentage_error(dep, rent_predictions)
#-10.9266

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

rent_predictions_test = rf1.predict(ind_test)
rf1_mse_test = mean_squared_error(dep_test, rent_predictions_test)
rf1_rmse_test = numpy.sqrt(rf1_mse_test)
rf1_rmse_test
#785.32
sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#63.73
mean_absolute_percentage_error(dep_test, rent_predictions_test)
#31.0233
mean_percentage_error(dep_test, rent_predictions_test)
#-14.8256

#Variable importance
feature_importances = rf1.feature_importances_
feature_importances
attributes = ind.columns
sorted(zip(feature_importances, attributes), reverse=True)

#(0.1211782159200295, ('weighted_rent_final', '0')),
# (0.096942915818886366, ('weighted_rent_final', '1')),
# (0.077622050943700352, ('weighted_rent_final', '3')),
# (0.071563187334169914, ('weighted_rent_final', '2')),
# (0.071055899824687851, ('weighted_rent_final', '4')),
# (0.064193739224425658, ('weighted_rent_final', '8')),
# (0.060561928604203764, ('weighted_rent_final', '9')),
# (0.048602166281900581, ('weighted_rent_final', '5')),
# (0.047228148077552504, ('weighted_rent_final', '6')),
# (0.037005604957307155, ('weighted_rent_final', '11')),
# (0.031168918777613373, ('weighted_rent_final', '10')),
# (0.024525516997398848, ('weighted_rent_final', '12')),
# (0.023079581235223388, ('weighted_rent_final', '14')),
# (0.020174737333943288, ('weighted_rent_final', '20')),
# (0.019658202843324611, ('weighted_rent_final', '7')),
# (0.019217531235445787, ('weighted_rent_final', '25')),
# (0.018182549654524403, ('weighted_rent_final', '22')),
# (0.016920456816272741, ('weighted_rent_final', '15')),
# (0.014400101296182018, ('weighted_rent_final', '29')),
# (0.014302220040432665, ('weighted_rent_final', '13')),
# (0.012183666747731107, ('weighted_rent_final', '18')),
# (0.011780181936034377, ('weighted_rent_final', '19')),
# (0.011716159920963979, ('weighted_rent_final', '23')),
# (0.011634157294653809, ('weighted_rent_final', '16')),
# (0.010789079642186103, ('weighted_rent_final', '24')),
# (0.010638502188609332, ('weighted_rent_final', '21')),
# (0.010326464495015959, ('weighted_rent_final', '17')),
# (0.009201650352472359, ('weighted_rent_final', '26')),
# (0.0080530666561512299, ('weighted_rent_final', '27')),
# (0.0060933975489569385, ('weighted_rent_final', '28'))

joblib.dump(rf1, "rf1.pkl")
#my_model_loaded = joblib.load("rf1.pkl")

#Random Forests for variable importance & GBM for rental predictions on independent variables with PCA scores

#Random Forests for variable importance & GBM for rental predictions on independent variables and zipcode variables