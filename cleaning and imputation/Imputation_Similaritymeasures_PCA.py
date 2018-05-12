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

#Imputation - Mean based

preimpute_finaldata = pandas.read_csv('/Users/aditilakra/Desktop/capstone/Dataaggregation/PreImpute_finaldata.csv')
preimpute_finaldata.drop('Unnamed: 0', axis=1, inplace=True)

#Check % of missing values across variables
Xmiss = (preimpute_finaldata.isnull().sum()/len(preimpute_finaldata.index)*100)
Xmiss1 = pandas.DataFrame(Xmiss)
Xmiss1.to_csv('/Users/aditilakra/Desktop/capstone/Dataaggregation/PreImpute_missing.csv')
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
#imputer.statistics_
#X.median().values
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

postimpute_finaldata.to_csv('/Users/aditilakra/Desktop/capstone/Dataaggregation/PostImpute_finaldata.csv')

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
                         'SE_T102_006',	'SE_T102_007',	'SE_T102_008',	'SE_T102_009',	'SE_T104_001',	'SE_T108_002',	'SE_T108_008',	
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
X1 = pca.fit_transform(X_std)
cumsum = numpy.cumsum(pca.explained_variance_ratio_)
d = numpy.argmax(cumsum >= 0.95) + 1
#21 components explain > 95% of variance
cumsum[:21]*100

#38.08894058,  58.09080848,  68.63781078,  74.0086987 ,
#        77.90061993,  80.56939532,  82.66277652,  84.46296414,
#        85.93325874,  87.22926256,  88.43604446,  89.60772089,
#        90.48867119,  91.2836458 ,  92.07462663,  92.74799745,
#        93.3550766 ,  93.90655614,  94.38071262,  94.81952278,  95.20675301

comp = [i+1 for i in range(131)]
Screeplot = pandas.DataFrame({'Cumulative Percentage Variance Explained': cumsum,'Components': comp})
Screeplotimage = Screeplot.plot(kind="scatter", x="Components", y="Cumulative Percentage Variance Explained",alpha=1)
plt.savefig('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/Screeplotimage.pdf')

pandas.DataFrame(pca.components_.T[:,0:21],X.columns).to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/PCA_Loadings_Zip.csv')

del(d)
del(cumsum)
del(X)
del(X_std)
del(Screeplot)
del(Screeplotimage)
del(comp)

#Extract 21 metafeatures and merge with main dataset to compute cosine similarity 

temp1 = pandas.DataFrame(X1[:,0:21])


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
                         'SE_T102_006',	'SE_T102_007',	'SE_T102_008',	'SE_T102_009',	'SE_T104_001',	'SE_T108_002',	'SE_T108_008',	
                         'SE_T113_002',	'SE_T113_011',	'SE_T128_002',	'SE_T128_003',	'SE_T128_004',	'SE_T128_005',	'SE_T128_006',	
                         'SE_T128_007',	'SE_T128_008',	'SE_T147_001',	'SE_T157_001',	'SE_T182_002',	'SE_T182_003',	'SE_T182_004',	
                         'SE_T182_005',	'SE_T182_006',	'SE_T182_007',	'SE_T191_002',	'SE_T191_003',	'SE_T191_004',	'SE_T191_005',	
                         'SE_T191_006',	'SE_T191_007',	'SE_T191_008',	'SE_T191_009',	'SE_T191_010',	'SE_T191_011',	'SE_T199_002',	
                         'SE_T199_003',	'SE_T199_004',	'SE_T199_005',	'SE_T199_006',	'SE_T199_007',	'SE_T211_001',	'SE_T211_002',	
                         'SE_T211_016',	'SE_T221_002',	'SE_T221_003',	'SE_T221_004',	'SE_T221_005',	'SE_T221_006',	'SE_T221_007',	
                         'SE_T221_008',	'SE_T221_009',	'SE_T221_010',	'SE_T221_011',	'SE_T221_012',	'SE_T227_001',	'SE_T235_002',	
                         'SE_T235_003',	'SE_T235_004',	'SE_T235_005',	'SE_T235_006',	'SE_T235_007'],axis=1)

postimpute_pca =  pandas.merge(temp2, temp1, left_index=True, right_index=True)

postimpute_pca.to_csv('/Users/aditilakra/Desktop/capstone/Dataaggregation/PostImpute_pca.csv')
del(temp1,temp2)                       

#Run Similarities across markets
columnstemp = pandas.DataFrame(postimpute_pca.columns)
Xtemp = postimpute_pca.drop(['address_final','zipcode_final','city_final','state_final','rent_final'],axis=1)
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
#cosine_similarity_stack_df = cosine_similarity_stack_df[cosine_similarity_stack_df['Index1_Rows']!="'Index'"]
cosine_similarity_stack_df['Index1_Rows'] = pandas.to_numeric(cosine_similarity_stack_df['Index1_Rows'])
cosine_similarity_stack_df['Index1_Columns'] = pandas.to_numeric(cosine_similarity_stack_df['Index1_Columns'])
cosine_similarity_stack_df = cosine_similarity_stack_df[cosine_similarity_stack_df['Index1_Rows'] != cosine_similarity_stack_df['Index1_Columns']]
cosine_similarity_stack_df = cosine_similarity_stack_df.sort_values(['Index1_Rows', 0], ascending=[True, False])
cosine_similarity_stack_df = cosine_similarity_stack_df.groupby('Index1_Rows').head(30)

Y = postimpute_finaldata.reset_index(drop=True)
Y['Index1'] = Y.index
Y['Index1'] = Y['Index1'].astype(int)
Y=Y[['Index1','rent_final']]
cosine_similarity_stack_df = pandas.merge(cosine_similarity_stack_df, Y, how='left', left_on=['Index1_Columns'],right_on=['Index1'])
cosine_similarity_stack_df = cosine_similarity_stack_df.drop(['Index1_x','Index1_y'], axis=1)

#k1 = cosine_similarity_stack_df.columns.to_series().groupby(cosine_similarity_stack_df.dtypes).groups
#k1

cosine_similarity_stack_df['Index1_Columns2'] = cosine_similarity_stack_df.groupby('Index1_Rows').cumcount()
cosine_similarity_stack_df['Index1_Columns2'] = cosine_similarity_stack_df['Index1_Columns2'].astype(str)
cosine_similarity_stack_df['Index1_Rows'] = cosine_similarity_stack_df['Index1_Rows'].astype(str)
cosine_similarity_stack_df.rename(columns={0: 'Cosine_sim'}, inplace=True)
cosine_similarity_stack_df['weighted_rent_final'] = cosine_similarity_stack_df['rent_final']*cosine_similarity_stack_df['Cosine_sim']

mlinput = pandas.pivot_table(cosine_similarity_stack_df,index=["Index1_Rows"],columns=["Index1_Columns2"],values=["rent_final","Cosine_sim","weighted_rent_final"])
mlinput.index = pandas.to_numeric(mlinput.index)
mlinput.sort_index(inplace=True)
mlinput.index = pandas.RangeIndex(start=0,stop=1386,step=1)
del(columnstemp,Xtemp_std,Xtemp,X,Y,cosine_similarity_df_stack,cs)

type(mlinput.index)
type(postimpute_pca.index)

mlinput = pandas.merge(mlinput, postimpute_pca[['rent_final','city_final']], how='left', left_index=True, right_index=True)

#Split data into train and test (30%) 
#Random Forests for variable importance & GBM for rental predictions on weighted rents

train_set, test_set = train_test_split(mlinput, test_size=0.3, random_state=42)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in split.split(mlinput, mlinput["city_final"]):
    strat_train_set = mlinput.loc[train_index]
    strat_test_set = mlinput.loc[test_index]

del(train_index,test_index,split)

postimpute_pca["city_final"].value_counts()/len(postimpute_pca)
#Austin         0.398268
#Chicago        0.295815
#Columbia       0.096681
#Newyork        0.059885
#Dayton         0.058442
#Eugene         0.045455
#Chattanooga    0.045455
train_set["city_final"].value_counts()/len(train_set)
test_set["city_final"].value_counts()/len(test_set)
(strat_train_set["city_final"].value_counts()/len(strat_train_set) - strat_test_set["city_final"].value_counts()/len(strat_test_set))/postimpute_pca["city_final"].value_counts()/len(postimpute_pca)*100
(train_set["city_final"].value_counts()/len(train_set) - test_set["city_final"].value_counts()/len(test_set))/postimpute_pca["city_final"].value_counts()/len(postimpute_pca)*100
#the difference in city-wise proportions is less in the stratified split

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

joblib.dump(rf1, "rf1.pkl")


ind.to_csv('ind.csv')
ind_test.to_csv('ind_test.csv')
dep.to_csv('dep.csv')
dep_test.to_csv('dep_test.csv')

#my_model_loaded = joblib.load("rf1.pkl")

#Random Forests for variable importance & GBM for rental predictions on independent variables with PCA scores

#Random Forests for variable importance & GBM for rental predictions on independent variables and zipcode variables