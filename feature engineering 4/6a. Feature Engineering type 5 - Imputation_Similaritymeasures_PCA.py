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
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import sklearn

#Set working directory
os.chdir('D:/Desktop/Capstone_Greystone/WorkingDirectory')
os.getcwd()

#Imputation - Median based
preimpute_finaldata = pandas.read_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Final Datasets/PreImpute_finaldata.csv')
preimpute_finaldata.drop('Unnamed: 0', axis=1, inplace=True)

#Check % of missing values across variables
Xmiss = (preimpute_finaldata.isnull().sum()/len(preimpute_finaldata.index)*100)
Xmiss1 = pandas.DataFrame(Xmiss)
Xmiss1.to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/PreImpute_missing.csv')
del(Xmiss)
del(Xmiss1)

#Impute Numerical variables with median
X = preimpute_finaldata[['SE_T001_001',	'SE_T002_002',	'SE_T004_002',	'SE_T004_003',	'SE_T009_002',	'SE_T009_003',	'SE_T009_004',	
                         'SE_T009_005',	'SE_T013_002',	'SE_T013_003',	'SE_T013_004',	'SE_T013_005',	'SE_T013_006',	'SE_T013_007',	
                         'SE_T013_008',	'SE_T017_001',	'SE_T017_002',	'SE_T017_007',	'SE_T021_001',	'SE_T028_002',	'SE_T028_003',	
                         'SE_T033_002',	'SE_T033_007',	'SE_T056_002',	'SE_T056_003',	'SE_T056_004',	'SE_T056_005',	'SE_T056_006',	
                         'SE_T056_007',	'SE_T056_008',	'SE_T056_009',	'SE_T056_010',	'SE_T056_011',	'SE_T056_012',	'SE_T056_013',	
                         'SE_T056_014',	'SE_T056_015',	'SE_T056_016',	'SE_T056_017',	'SE_T067_002',	'SE_T067_003',	'SE_T083_001',	
                         'SE_T094_002',	'SE_T094_003',	'SE_T096_002',	'SE_T096_003',	'SE_T096_004',	'SE_T097_002',	'SE_T097_005',	
                         'SE_T097_006',	'SE_T097_007',	'SE_T097_008',	'SE_T097_009',	'SE_T097_010',	'SE_T097_011',	'SE_T097_012',	
                         'SE_T098_001',	'SE_T100_002',	'SE_T100_003',	'SE_T100_004',	'SE_T100_005',	'SE_T100_006',	'SE_T100_007',	
                         'SE_T100_008',	'SE_T100_009',	'SE_T100_010',	'SE_T102_002',	'SE_T102_003',	'SE_T102_004',	'SE_T102_005',	
                         'SE_T102_006',	'SE_T102_007',	'SE_T102_008',	'SE_T102_009',	'SE_T104_001',	'SE_T108_002',	'SE_T108_008',	
                         'SE_T113_002',	'SE_T113_011',	'SE_T128_002',	'SE_T128_003',	'SE_T128_004',	'SE_T128_005',	'SE_T128_006',	
                         'SE_T128_007',	'SE_T128_008',	'SE_T147_001',	'SE_T157_001',	'SE_T182_002',	'SE_T182_003',	'SE_T182_004',	
                         'SE_T182_005',	'SE_T182_006',	'SE_T182_007',	'SE_T191_002',	'SE_T191_003',	'SE_T191_004',	'SE_T191_005',	
                         'SE_T191_006',	'SE_T191_007',	'SE_T191_008',	'SE_T191_009',	'SE_T191_010',	'SE_T191_011',	'SE_T199_002',	
                         'SE_T199_003',	'SE_T199_004',	'SE_T199_005',	'SE_T199_006',	'SE_T199_007',	'SE_T211_001',	'SE_T211_002',	
                         'SE_T211_016',	'SE_T221_002',	'SE_T221_003',	'SE_T221_004',	'SE_T221_005',	'SE_T221_006',	'SE_T221_007',	
                         'SE_T221_008',	'SE_T221_009',	'SE_T221_010',	'SE_T221_011',	'SE_T221_012',	'SE_T227_001',	'SE_T235_002',	
                         'SE_T235_003',	'SE_T235_004',	'SE_T235_005',	'SE_T235_006',	'SE_T235_007',	'bedrooms_final',	
                         'bathrooms_final',	'sqft_final',	'age_yrs_final']]

imputer = Imputer(strategy="median")
imputer.fit(X)
X_tran = imputer.transform(X)
X_imp = pandas.DataFrame(X_tran, columns=X.columns)
preimpute_finaldata.drop(['SE_T001_001',	'SE_T002_002',	'SE_T004_002',	'SE_T004_003',	'SE_T009_002',	'SE_T009_003',	'SE_T009_004',	
                          'SE_T009_005',	'SE_T013_002',	'SE_T013_003',	'SE_T013_004',	'SE_T013_005',	'SE_T013_006',	'SE_T013_007',	
                          'SE_T013_008',	'SE_T017_001',	'SE_T017_002',	'SE_T017_007',	'SE_T021_001',	'SE_T028_002',	'SE_T028_003',	
                          'SE_T033_002',	'SE_T033_007',	'SE_T056_002',	'SE_T056_003',	'SE_T056_004',	'SE_T056_005',	'SE_T056_006',	
                          'SE_T056_007',	'SE_T056_008',	'SE_T056_009',	'SE_T056_010',	'SE_T056_011',	'SE_T056_012',	'SE_T056_013',	
                          'SE_T056_014',	'SE_T056_015',	'SE_T056_016',	'SE_T056_017',	'SE_T067_002',	'SE_T067_003',	'SE_T083_001',	
                          'SE_T094_002',	'SE_T094_003',	'SE_T096_002',	'SE_T096_003',	'SE_T096_004',	'SE_T097_002',	'SE_T097_005',	
                          'SE_T097_006',	'SE_T097_007',	'SE_T097_008',	'SE_T097_009',	'SE_T097_010',	'SE_T097_011',	'SE_T097_012',	
                          'SE_T098_001',	'SE_T100_002',	'SE_T100_003',	'SE_T100_004',	'SE_T100_005',	'SE_T100_006',	'SE_T100_007',	
                          'SE_T100_008',	'SE_T100_009',	'SE_T100_010',	'SE_T102_002',	'SE_T102_003',	'SE_T102_004',	'SE_T102_005',	
                          'SE_T102_006',	'SE_T102_007',	'SE_T102_008',	'SE_T102_009',	'SE_T104_001',	'SE_T108_002',	'SE_T108_008',	
                          'SE_T113_002',	'SE_T113_011',	'SE_T128_002',	'SE_T128_003',	'SE_T128_004',	'SE_T128_005',	'SE_T128_006',	
                          'SE_T128_007',	'SE_T128_008',	'SE_T147_001',	'SE_T157_001',	'SE_T182_002',	'SE_T182_003',	'SE_T182_004',	
                          'SE_T182_005',	'SE_T182_006',	'SE_T182_007',	'SE_T191_002',	'SE_T191_003',	'SE_T191_004',	'SE_T191_005',	
                          'SE_T191_006',	'SE_T191_007',	'SE_T191_008',	'SE_T191_009',	'SE_T191_010',	'SE_T191_011',	'SE_T199_002',	
                          'SE_T199_003',	'SE_T199_004',	'SE_T199_005',	'SE_T199_006',	'SE_T199_007',	'SE_T211_001',	'SE_T211_002',	
                          'SE_T211_016',	'SE_T221_002',	'SE_T221_003',	'SE_T221_004',	'SE_T221_005',	'SE_T221_006',	'SE_T221_007',	
                          'SE_T221_008',	'SE_T221_009',	'SE_T221_010',	'SE_T221_011',	'SE_T221_012',	'SE_T227_001',	'SE_T235_002',	
                          'SE_T235_003',	'SE_T235_004',	'SE_T235_005',	'SE_T235_006',	'SE_T235_007',	'bedrooms_final',	
                          'bathrooms_final',	'sqft_final',	'age_yrs_final'],axis=1,inplace=True)

preimpute_finaldata = pandas.merge(preimpute_finaldata, X_imp, left_index=True, right_index=True)      

del(X)
del(X_tran)
del(X_imp)

#Impute binary 1-0 variables to 0 if Nan
postimpute_finaldata = preimpute_finaldata.fillna(0)

#Correlation
X = postimpute_finaldata[['SE_T001_001',	'SE_T002_002',	'SE_T004_002',	'SE_T004_003',	'SE_T009_002',	'SE_T009_003',	'SE_T009_004',	
                         'SE_T009_005',	'SE_T013_002',	'SE_T013_003',	'SE_T013_004',	'SE_T013_005',	'SE_T013_006',	'SE_T013_007',	
                         'SE_T013_008',	'SE_T017_001',	'SE_T017_002',	'SE_T017_007',	'SE_T021_001',	'SE_T028_002',	'SE_T028_003',	
                         'SE_T033_002',	'SE_T033_007',	'SE_T056_002',	'SE_T056_003',	'SE_T056_004',	'SE_T056_005',	'SE_T056_006',	
                         'SE_T056_007',	'SE_T056_008',	'SE_T056_009',	'SE_T056_010',	'SE_T056_011',	'SE_T056_012',	'SE_T056_013',	
                         'SE_T056_014',	'SE_T056_015',	'SE_T056_016',	'SE_T056_017',	'SE_T067_002',	'SE_T067_003',	'SE_T083_001',	
                         'SE_T094_002',	'SE_T094_003',	'SE_T096_002',	'SE_T096_003',	'SE_T096_004',	'SE_T097_002',	'SE_T097_005',	
                         'SE_T097_006',	'SE_T097_007',	'SE_T097_008',	'SE_T097_009',	'SE_T097_010',	'SE_T097_011',	'SE_T097_012',	
                         'SE_T098_001',	'SE_T100_002',	'SE_T100_003',	'SE_T100_004',	'SE_T100_005',	'SE_T100_006',	'SE_T100_007',	
                         'SE_T100_008',	'SE_T100_009',	'SE_T100_010',	'SE_T102_002',	'SE_T102_003',	'SE_T102_004',	'SE_T102_005',	
                         'SE_T102_006',	'SE_T102_007',	'SE_T102_008',	'SE_T102_009',	'SE_T104_001',	'SE_T108_002',	'SE_T108_008',	
                         'SE_T113_002',	'SE_T113_011',	'SE_T128_002',	'SE_T128_003',	'SE_T128_004',	'SE_T128_005',	'SE_T128_006',	
                         'SE_T128_007',	'SE_T128_008',	'SE_T147_001',	'SE_T157_001',	'SE_T182_002',	'SE_T182_003',	'SE_T182_004',	
                         'SE_T182_005',	'SE_T182_006',	'SE_T182_007',	'SE_T191_002',	'SE_T191_003',	'SE_T191_004',	'SE_T191_005',	
                         'SE_T191_006',	'SE_T191_007',	'SE_T191_008',	'SE_T191_009',	'SE_T191_010',	'SE_T191_011',	'SE_T199_002',	
                         'SE_T199_003',	'SE_T199_004',	'SE_T199_005',	'SE_T199_006',	'SE_T199_007',	'SE_T211_001',	'SE_T211_002',	
                         'SE_T211_016',	'SE_T221_002',	'SE_T221_003',	'SE_T221_004',	'SE_T221_005',	'SE_T221_006',	'SE_T221_007',	
                         'SE_T221_008',	'SE_T221_009',	'SE_T221_010',	'SE_T221_011',	'SE_T221_012',	'SE_T227_001',	'SE_T235_002',	
                         'SE_T235_003',	'SE_T235_004',	'SE_T235_005',	'SE_T235_006',	'SE_T235_007',	'bedrooms_final',	
                         'bathrooms_final',	'sqft_final', 'age_yrs_final', 'units_final','rent_final']]

corr_matrix = X.corr()
corr_matrix["rent_final"].sort_values(ascending=False).to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\Analysis\correlation_7markets.csv')
del(X)

#Run PCA on Zip code variables
X = postimpute_finaldata[['SE_T001_001',	'SE_T002_002',	'SE_T004_002',	'SE_T004_003',	'SE_T009_002',	'SE_T009_003',	'SE_T009_004',	
                         'SE_T009_005',	'SE_T013_002',	'SE_T013_003',	'SE_T013_004',	'SE_T013_005',	'SE_T013_006',	'SE_T013_007',	
                         'SE_T013_008',	'SE_T017_001',	'SE_T017_002',	'SE_T017_007',	'SE_T021_001',	'SE_T028_002',	'SE_T028_003',	
                         'SE_T033_002',	'SE_T033_007',	'SE_T056_002',	'SE_T056_003',	'SE_T056_004',	'SE_T056_005',	'SE_T056_006',	
                         'SE_T056_007',	'SE_T056_008',	'SE_T056_009',	'SE_T056_010',	'SE_T056_011',	'SE_T056_012',	'SE_T056_013',	
                         'SE_T056_014',	'SE_T056_015',	'SE_T056_016',	'SE_T056_017',	'SE_T067_002',	'SE_T067_003',	'SE_T083_001',	
                         'SE_T094_002',	'SE_T094_003',	'SE_T096_002',	'SE_T096_003',	'SE_T096_004',	'SE_T097_002',	'SE_T097_005',	
                         'SE_T097_006',	'SE_T097_007',	'SE_T097_008',	'SE_T097_009',	'SE_T097_010',	'SE_T097_011',	'SE_T097_012',	
                         'SE_T098_001',	'SE_T100_002',	'SE_T100_003',	'SE_T100_004',	'SE_T100_005',	'SE_T100_006',	'SE_T100_007',	
                         'SE_T100_008',	'SE_T100_009',	'SE_T100_010',	'SE_T102_002',	'SE_T102_003',	'SE_T102_004',	'SE_T102_005',	
                         'SE_T102_006',	'SE_T102_007',	'SE_T102_008',	'SE_T102_009',	               	'SE_T108_002',	'SE_T108_008',	
                         'SE_T113_002',	'SE_T113_011',	'SE_T128_002',	'SE_T128_003',	'SE_T128_004',	'SE_T128_005',	'SE_T128_006',	
                         'SE_T128_007',	'SE_T128_008',	'SE_T147_001',	'SE_T157_001',	'SE_T182_002',	'SE_T182_003',	'SE_T182_004',	
                         'SE_T182_005',	'SE_T182_006',	'SE_T182_007',	'SE_T191_002',	'SE_T191_003',	'SE_T191_004',	'SE_T191_005',	
                         'SE_T191_006',	'SE_T191_007',	'SE_T191_008',	'SE_T191_009',	'SE_T191_010',	'SE_T191_011',	'SE_T199_002',	
                         'SE_T199_003',	'SE_T199_004',	'SE_T199_005',	'SE_T199_006',	'SE_T199_007',	'SE_T211_001',	'SE_T211_002',	
                         'SE_T211_016',	'SE_T221_002',	'SE_T221_003',	'SE_T221_004',	'SE_T221_005',	'SE_T221_006',	'SE_T221_007',	
                         'SE_T221_008',	'SE_T221_009',	'SE_T221_010',	'SE_T221_011',	'SE_T221_012',	'SE_T227_001',	'SE_T235_002',	
                         'SE_T235_003',	'SE_T235_004',	'SE_T235_005',	'SE_T235_006',	'SE_T235_007']]

sc = StandardScaler(with_mean=True, with_std=True)
X_std = sc.fit_transform(X)
pca = PCA()
X1_wo_MGR = pca.fit_transform(X_std)
cumsum = numpy.cumsum(pca.explained_variance_ratio_)
d = numpy.argmax(cumsum >= 0.95) + 1
#21 components explain > 95% of variance
cumsum[:21]*100
#38.37410201,  58.13526133,  68.7098604 ,  74.11852809,
#        77.95518575,  80.64009067,  82.73987251,  84.55356075,
#        86.01945653,  87.32324768,  88.51651005,  89.67531689,
#        90.56295217,  91.36099089,  92.12143593,  92.78683837,
#        93.37692354,  93.92037011,  94.39642447,  94.8386056 ,  95.2239016


comp = [i+1 for i in range(130)]
Screeplot = pandas.DataFrame({'Cumulative Percentage Variance Explained': cumsum,'Components': comp})
Screeplotimage = Screeplot.plot(kind="scatter", x="Components", y="Cumulative Percentage Variance Explained",alpha=1)
plt.savefig('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/Screeplotimage_wo_MGR.pdf')

pandas.DataFrame(pca.components_.T[:,0:21],X.columns).to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/PCA_Loadings_Zip_wo_MGR.csv')

del(d)
del(cumsum)
del(X)
del(X_std)
del(Screeplot)
del(Screeplotimage)
del(comp)

#Extract 21 metafeatures and merge with main dataset to compute cosine similarity 
temp1 = pandas.DataFrame(X1_wo_MGR[:,0:21])

temp1.columns = ['Compscore1', 'Compscore2', 'Compscore3', 'Compscore4', 'Compscore5', 'Compscore6', 'Compscore7', 'Compscore8',
                 'Compscore9', 'Compscore10', 'Compscore11', 'Compscore12', 'Compscore13', 'Compscore14', 'Compscore15', 'Compscore16',
                 'Compscore17', 'Compscore18', 'Compscore19', 'Compscore20', 'Compscore21']

temp2 = postimpute_finaldata.drop(['SE_T001_001',	'SE_T002_002',	'SE_T004_002',	'SE_T004_003',	'SE_T009_002',	'SE_T009_003',	'SE_T009_004',	
                         'SE_T009_005',	'SE_T013_002',	'SE_T013_003',	'SE_T013_004',	'SE_T013_005',	'SE_T013_006',	'SE_T013_007',	
                         'SE_T013_008',	'SE_T017_001',	'SE_T017_002',	'SE_T017_007',	'SE_T021_001',	'SE_T028_002',	'SE_T028_003',	
                         'SE_T033_002',	'SE_T033_007',	'SE_T056_002',	'SE_T056_003',	'SE_T056_004',	'SE_T056_005',	'SE_T056_006',	
                         'SE_T056_007',	'SE_T056_008',	'SE_T056_009',	'SE_T056_010',	'SE_T056_011',	'SE_T056_012',	'SE_T056_013',	
                         'SE_T056_014',	'SE_T056_015',	'SE_T056_016',	'SE_T056_017',	'SE_T067_002',	'SE_T067_003',	'SE_T083_001',	
                         'SE_T094_002',	'SE_T094_003',	'SE_T096_002',	'SE_T096_003',	'SE_T096_004',	'SE_T097_002',	'SE_T097_005',	
                         'SE_T097_006',	'SE_T097_007',	'SE_T097_008',	'SE_T097_009',	'SE_T097_010',	'SE_T097_011',	'SE_T097_012',	
                         'SE_T098_001',	'SE_T100_002',	'SE_T100_003',	'SE_T100_004',	'SE_T100_005',	'SE_T100_006',	'SE_T100_007',	
                         'SE_T100_008',	'SE_T100_009',	'SE_T100_010',	'SE_T102_002',	'SE_T102_003',	'SE_T102_004',	'SE_T102_005',	
                         'SE_T102_006',	'SE_T102_007',	'SE_T102_008',	'SE_T102_009',	              	'SE_T108_002',	'SE_T108_008',	
                         'SE_T113_002',	'SE_T113_011',	'SE_T128_002',	'SE_T128_003',	'SE_T128_004',	'SE_T128_005',	'SE_T128_006',	
                         'SE_T128_007',	'SE_T128_008',	'SE_T147_001',	'SE_T157_001',	'SE_T182_002',	'SE_T182_003',	'SE_T182_004',	
                         'SE_T182_005',	'SE_T182_006',	'SE_T182_007',	'SE_T191_002',	'SE_T191_003',	'SE_T191_004',	'SE_T191_005',	
                         'SE_T191_006',	'SE_T191_007',	'SE_T191_008',	'SE_T191_009',	'SE_T191_010',	'SE_T191_011',	'SE_T199_002',	
                         'SE_T199_003',	'SE_T199_004',	'SE_T199_005',	'SE_T199_006',	'SE_T199_007',	'SE_T211_001',	'SE_T211_002',	
                         'SE_T211_016',	'SE_T221_002',	'SE_T221_003',	'SE_T221_004',	'SE_T221_005',	'SE_T221_006',	'SE_T221_007',	
                         'SE_T221_008',	'SE_T221_009',	'SE_T221_010',	'SE_T221_011',	'SE_T221_012',	'SE_T227_001',	'SE_T235_002',	
                         'SE_T235_003',	'SE_T235_004',	'SE_T235_005',	'SE_T235_006',	'SE_T235_007'],axis=1)

postimpute_pca =  pandas.merge(temp2, temp1, left_index=True, right_index=True)
del(temp1,temp2)                       

#Run Similarities across markets
columnstemp = pandas.DataFrame(postimpute_pca.columns)
Xtemp = postimpute_pca.drop(['address_final','zipcode_final','city_final','state_final','rent_final','SE_T104_001'],axis=1)
sc = StandardScaler(with_mean=True, with_std=True)
Xtemp_std = sc.fit_transform(Xtemp)
X = Xtemp_std.copy()
cs = cosine_similarity(X)
cosine_similarity_df = pandas.DataFrame(cs)
cosine_similarity_df_stack = cosine_similarity_df.stack()
cosine_similarity_stack_df = pandas.DataFrame(cosine_similarity_df_stack)
cosine_similarity_stack_df['Index1'] = cosine_similarity_stack_df.index
cosine_similarity_stack_df['Index1'] = cosine_similarity_stack_df['Index1'].astype(str)
cosine_similarity_stack_df['Index1_Rows'], cosine_similarity_stack_df['Index1_Columns'] = cosine_similarity_stack_df['Index1'].str.split(',', 1).str
cosine_similarity_stack_df['Index1_Rows'] = cosine_similarity_stack_df['Index1_Rows'].str.replace(r"\(","")
cosine_similarity_stack_df['Index1_Columns'] = cosine_similarity_stack_df['Index1_Columns'].str.replace(r"\)","")
cosine_similarity_stack_df['Index1_Rows'] = pandas.to_numeric(cosine_similarity_stack_df['Index1_Rows'])
cosine_similarity_stack_df['Index1_Columns'] = pandas.to_numeric(cosine_similarity_stack_df['Index1_Columns'])
cosine_similarity_stack_df = cosine_similarity_stack_df[cosine_similarity_stack_df['Index1_Rows'] != cosine_similarity_stack_df['Index1_Columns']]
cosine_similarity_stack_df = cosine_similarity_stack_df.sort_values(['Index1_Rows', 0], ascending=[True, False])
cosine_similarity_stack_df = cosine_similarity_stack_df.groupby('Index1_Rows').head(30)
Y = postimpute_finaldata.reset_index(drop=True)
Y['Index1'] = Y.index
Y['Index1'] = Y['Index1'].astype(int)
Y=Y[['Index1','rent_final','SE_T104_001']]
cosine_similarity_stack_df = pandas.merge(cosine_similarity_stack_df, Y, how='left', left_on=['Index1_Columns'],right_on=['Index1'])
cosine_similarity_stack_df = cosine_similarity_stack_df.drop(['Index1_x','Index1_y'], axis=1)
cosine_similarity_stack_df['Index1_Columns2'] = cosine_similarity_stack_df.groupby('Index1_Rows').cumcount()
cosine_similarity_stack_df['Index1_Columns2'] = cosine_similarity_stack_df['Index1_Columns2'].astype(str)
cosine_similarity_stack_df['Index1_Rows'] = cosine_similarity_stack_df['Index1_Rows'].astype(str)
cosine_similarity_stack_df.rename(columns={0: 'Cosine_sim'}, inplace=True)
cosine_similarity_stack_df['weighted_rent_final'] = cosine_similarity_stack_df['rent_final']*cosine_similarity_stack_df['Cosine_sim']
mlinput_wo_MGR = pandas.pivot_table(cosine_similarity_stack_df,index=["Index1_Rows"],columns=["Index1_Columns2"],values=["rent_final","Cosine_sim","weighted_rent_final","SE_T104_001"])
mlinput_wo_MGR.index = pandas.to_numeric(mlinput_wo_MGR.index)
mlinput_wo_MGR.sort_index(inplace=True)
mlinput_wo_MGR.index = pandas.RangeIndex(start=0,stop=1386,step=1)

del(columnstemp,Xtemp_std,Xtemp,X,Y,cosine_similarity_df_stack,cs)

mlinput_wo_MGR = pandas.merge(mlinput_wo_MGR, postimpute_pca[['rent_final','city_final','SE_T104_001']], how='left', left_index=True, right_index=True)

#Split data into train and test (30%) 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in split.split(mlinput_wo_MGR, mlinput_wo_MGR["city_final"]):
    strat_train_set_wo_MGR1 = mlinput_wo_MGR.loc[train_index]
    strat_test_set_wo_MGR1 = mlinput_wo_MGR.loc[test_index]

del(train_index,test_index,split)
postimpute_pca["city_final"].value_counts()/len(postimpute_pca)
#Austin         0.398268
#Chicago        0.295815
#Columbia       0.096681
#Newyork        0.059885
#Dayton         0.058442
#Eugene         0.045455
#Chattanooga    0.045455

(strat_train_set_wo_MGR1["city_final"].value_counts()/len(strat_train_set_wo_MGR1) - strat_test_set_wo_MGR1["city_final"].value_counts()/len(strat_test_set_wo_MGR1))/postimpute_pca["city_final"].value_counts()/len(postimpute_pca)*100
#Austin        -1.438188e-07
#Chattanooga   -3.576035e-07
#Chicago        3.576035e-08
#Columbia       4.056397e-07
#Dayton         9.536092e-07
#Eugene        -3.576035e-07
#Newyork       -2.628170e-07
#the difference in city-wise proportions is negligible

strat_train_set_wo_MGR1.to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Final Datasets/strat_train_set_wo_MGR1.csv')
strat_test_set_wo_MGR1.to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Final Datasets/strat_test_set_wo_MGR1.csv')
