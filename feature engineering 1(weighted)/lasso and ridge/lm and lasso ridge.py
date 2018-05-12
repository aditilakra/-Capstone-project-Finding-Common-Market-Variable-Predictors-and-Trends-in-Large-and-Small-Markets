import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

strat_train_set= pd.read_csv("/Users/aditilakra/Desktop/capstone/Stratified_data /train.csv")
strat_train_set.drop('Unnamed: 0', axis=1, inplace=True)
strat_test_set= pd.read_csv("/Users/aditilakra/Desktop/capstone/Stratified_data /test.csv")

strat_test_set.drop('Unnamed: 0', axis=1, inplace=True)


#Support Vector Regression
ind = strat_train_set[["('weighted_rent_final', '0')",  "('weighted_rent_final', '1')",
       "('weighted_rent_final', '10')", "('weighted_rent_final', '11')",
       "('weighted_rent_final', '12')", "('weighted_rent_final', '13')",
       "('weighted_rent_final', '14')", "('weighted_rent_final', '15')",
       "('weighted_rent_final', '16')", "('weighted_rent_final', '17')",
       "('weighted_rent_final', '18')", "('weighted_rent_final', '19')",
        "('weighted_rent_final', '2')", "('weighted_rent_final', '20')",
       "('weighted_rent_final', '21')", "('weighted_rent_final', '22')",
       "('weighted_rent_final', '23')", "('weighted_rent_final', '24')",
       "('weighted_rent_final', '25')", "('weighted_rent_final', '26')",
      "('weighted_rent_final', '27')", "('weighted_rent_final', '28')",
       "('weighted_rent_final', '29')",  "('weighted_rent_final', '3')",
        "('weighted_rent_final', '4')",  "('weighted_rent_final', '5')",
        "('weighted_rent_final', '6')",  "('weighted_rent_final', '7')",
        "('weighted_rent_final', '8')",  "('weighted_rent_final', '9')"]].copy()

dep = strat_train_set['rent_final'].copy()


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


#test error
ind_test = strat_test_set[["('weighted_rent_final', '0')",  "('weighted_rent_final', '1')",
       "('weighted_rent_final', '10')", "('weighted_rent_final', '11')",
       "('weighted_rent_final', '12')", "('weighted_rent_final', '13')",
       "('weighted_rent_final', '14')", "('weighted_rent_final', '15')",
       "('weighted_rent_final', '16')", "('weighted_rent_final', '17')",
       "('weighted_rent_final', '18')", "('weighted_rent_final', '19')",
        "('weighted_rent_final', '2')", "('weighted_rent_final', '20')",
       "('weighted_rent_final', '21')", "('weighted_rent_final', '22')",
       "('weighted_rent_final', '23')", "('weighted_rent_final', '24')",
       "('weighted_rent_final', '25')", "('weighted_rent_final', '26')",
      "('weighted_rent_final', '27')", "('weighted_rent_final', '28')",
       "('weighted_rent_final', '29')",  "('weighted_rent_final', '3')",
        "('weighted_rent_final', '4')",  "('weighted_rent_final', '5')",
        "('weighted_rent_final', '6')",  "('weighted_rent_final', '7')",
        "('weighted_rent_final', '8')",  "('weighted_rent_final', '9')"]].copy()
dep_test = strat_test_set['rent_final'].copy()

ind_test_scaled = sc.transform(ind_test) 



##linear model

import statsmodels.api as sm



# Note the difference in argument order
results= sm.OLS(dep, ind).fit()
predictions = results.predict(ind) # make the predictions by the model

# Print out the statistics
results.summary()
# %load data/signif.py
def signif(fit_result_table):
    coef_table = pd.DataFrame(fit_result_table[1].data,  columns =  ['variable','coef','std err','z','P>|z|','[0.025','0.975]']).\
                                                                    set_index('variable')
    # first row is columns names, drop it
    coef_table = coef_table.drop([''], axis = 0)
    coef_table.iloc[:,3] = coef_table.iloc[:,3].astype(np.float32)

        # define a significance column similar to R
    coef_table[' '] = ['ooo' if i<0.001 else 'oo' if i>0.001 and i<=.01\
                  else 'o'  if i>.01 and i <=.05\
                  else '.'  if i>.05 and i <=.1 else ' ' for i in coef_table.iloc[:,3]]
    return(coef_table)  

 signif(results.summary())

print('F statistic:   ',results.fvalue)
print('F stat pvalue: ',results.f_pvalue)
print('r squared:     ',results.rsquared)
print('r squared adj: ',results.rsquared_adj)
print('residuals df:  ',results.df_resid)
print('SST:           ',results.mse_total*(results.df_resid + results.df_model))
print('SSE:           ',results.mse_resid*results.df_resid)
print('SSM:           ',results.mse_model*results.df_model)
print('resid std err: ',results.mse_resid**.5)

F statistic:    199.56781615998705
F stat pvalue:  0.0
r squared:      0.8642997949464961
r squared adj:  0.8599689373384056
residuals df:   940.0
SST:            4274836588.8559027
SSE:            580096201.6779672
SSM:            3694740387.1779356
resid std err:  785.5721601522982

import seaborn as sns
corr = ind.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

sns.set(style="white") # options include white, dark, whitegrid, darkgrid, ticks
sns.heatmap(corr, mask = mask)



# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(ind.values, i) for i in range(ind.shape[1])]
vif["features"] = ind.columns

vif.round(1)

def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.mean(numpy.abs(y_pred - y_true) / y_true) * 100

def mean_percentage_error(y_true, y_pred): 
    return numpy.mean((y_pred - y_true) / y_true) * 100


#Ridge regression

alphas = 10**np.linspace(10,-2,100)*0.5
alphas

ridge = Ridge(max_iter=10000,normalize=True)
coefs = []
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(ind, dep)
    coefs.append(ridge.coef_)
np.shape(coefs)


ridgecv = RidgeCV(alphas=alphas, scoring='mean_squared_error', normalize=True)
ridgecv.fit(ind, dep)
ridgecv.alpha_
#0.438745

ridge2 = Ridge(alpha=0.438745, normalize=True)
ridge2.fit(ind, dep) # Fit a ridge regression on the training data
pred1 = ridge2.predict(ind) # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index=ind.columns)) # Print coefficients

pd.Series(ridge2.coef_, index=ind.columns)



ridge2_mse = mean_squared_error(dep, pred1)
ridge2_rmse = numpy.sqrt(ridge2_mse)
ridge2_rmse
#778.888
sklearn.metrics.r2_score(dep,pred1)*100
#58.473




#test errors
temp_test =ridge2.predict(ind_test)
ridge2_mse_test = mean_squared_error(dep_test, temp_test)
ridge2_rmse_test = numpy.sqrt(ridge2_mse_test)
ridge2_rmse_test
#811.543
sklearn.metrics.r2_score(dep_test, temp_test)*100
#61.26
mean_absolute_percentage_error(dep_test, temp_test)
#32.201
mean_percentage_error(dep_test, temp_test)
#-14.24

ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

##LASSO REGRESSION

lasso = Lasso(max_iter=10000, normalize=True)
coefs = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(ind, dep)
    coefs.append(lasso.coef_)


lassocv = LassoCV(alphas=alphas,  cv=10, max_iter=100000, normalize=True)
lassocv.fit(ind, dep)
lassocv.alpha_
#1.3280

lasso2 = Lasso(alpha=1.3280, normalize=True)
lasso2.fit(ind, dep) # Fit a ridge regression on the training data
pred2 = lasso2.predict(ind) # Use this model to predict the test data
print(pd.Series(lasso2.coef_, index=ind.columns) )# Print coefficients

pd.Series(lasso2.coef_, index=ind.columns)
#Cross-validated Train errors



lasso2_mse = mean_squared_error(dep, pred2)
lasso2_rmse = numpy.sqrt(lasso2_mse)
lasso2_rmse
sklearn.metrics.r2_score(dep,pred2)*100
#58.64

###test errors
temp_test =lasso2.predict(ind_test)
lasso2_mse_test = mean_squared_error(dep_test, temp_test)
lasso2_rmse_test = numpy.sqrt(lasso2_mse_test)
lasso2_rmse_test
#798.49
sklearn.metrics.r2_score(dep_test, temp_test)*100
#62.5
mean_absolute_percentage_error(dep_test, temp_test)
#31.15
mean_percentage_error(dep_test, temp_test)
#-14.06


ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')