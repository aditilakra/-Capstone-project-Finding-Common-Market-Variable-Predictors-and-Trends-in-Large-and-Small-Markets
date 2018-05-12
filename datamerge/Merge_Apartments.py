#Merge Apartments.com Data to DSCompsZip2016

import pandas
import numpy
import os
import csv

#Use DSCompsZip2016 data

#dscompszip = pandas.read_csv('D:/Desktop/Capstone_Greystone/Data aggregation/eugene_dscompszip.csv')
#dscompszip = pandas.read_csv('D:/Desktop/Capstone_Greystone/Data aggregation/columbia_dscompszip.csv')
#dscompszip = pandas.read_csv('D:/Desktop/Capstone_Greystone/Data aggregation/dayton_dscompszip.csv')
#dscompszip = pandas.read_csv('D:/Desktop/Capstone_Greystone/Data aggregation/chattanooga_dscompszip.csv')
#dscompszip = pandas.read_csv('D:/Desktop/Capstone_Greystone/Data aggregation/austin_dscompszip.csv')
#dscompszip = pandas.read_csv('D:/Desktop/Capstone_Greystone/Data aggregation/chicago_dscompszip.csv')
dscompszip = pandas.read_csv('D:/Desktop/Capstone_Greystone/Data aggregation/newyork_dscompszip.csv')

dscompszip.drop('Unnamed: 0', axis=1, inplace=True)

#Read Aparqtments.com Data
#apt = pandas.read_excel('D:/Desktop/Capstone_Greystone/Data aggregation/Apartments.com Data/EUGENE.xlsx')
#apt = pandas.read_excel('D:/Desktop/Capstone_Greystone/Data aggregation/Apartments.com Data/columbia.xlsx')
#apt = pandas.read_excel('D:/Desktop/Capstone_Greystone/Data aggregation/Apartments.com Data/dayton.xlsx')
#apt = pandas.read_excel('D:/Desktop/Capstone_Greystone/Data aggregation/Apartments.com Data/Chattanooga.xlsx')
#apt = pandas.read_excel('D:/Desktop/Capstone_Greystone/Data aggregation/Apartments.com Data/Austin.xlsx')
#apt = pandas.read_excel('D:/Desktop/Capstone_Greystone/Data aggregation/Apartments.com Data/Chicago.xlsx')
apt = pandas.read_excel('D:/Desktop/Capstone_Greystone/Data aggregation/Apartments.com Data/Newyork.xlsx')

#Check for duplicates - Property Name, Address

#pandas.value_counts(apt['Property Name']).to_frame().reset_index()
#pandas.value_counts(apt['Address']).to_frame().reset_index()
#pandas.value_counts(apt['bedroom']).to_frame().reset_index()
#'Monthly Fees'	'One Time Fees'	'Pet Policy'	'Parking'	'Gym'	'Kitchen'	'Amenities'	'Features'	'Living Space'	'Lease Info'	'Services'	'Indoor Info'	'Outdoor Info'
apt_cat = apt.drop_duplicates(subset=["Address","City","State","Zipcode"])
g1 = apt_cat.columns.to_series().groupby(apt_cat.dtypes).groups
g1

apt_cat.drop(['Property Name', 'City', 'State','Zipcode', 'bedroom ','bathroom', 'rent', 'deposit', 'size', 'year built', 'year renovated','units'], axis=1, inplace=True)
#Roll up to address along with bedrooms, bathrooms, count of units, sqft, year built

aptv1 = pandas.pivot_table(apt,index=["Address","City","State","Zipcode"],values=["bedroom ","units","year built","size","bathroom","rent"],aggfunc={"bedroom ":numpy.mean,"units":numpy.mean,"year built":numpy.min,"size":numpy.mean,"bathroom":numpy.mean,"rent":numpy.mean})
aptv1['Index1'] = aptv1.index
#print (aptv1.dtypes)

#for col in aptv1:
#                print (aptv1[col].apply(type))
 #Index1 is tuple
aptv2 = aptv1['Index1'].apply(pandas.Series) 
aptv2['Index1'] = aptv2.index
aptv1 = pandas.merge(aptv1, aptv2, how='inner', left_on=['Index1'],right_on=['Index1'])
aptv1.drop('Index1', axis=1, inplace=True)
aptv1.rename(columns={0: 'Address', 1: 'City', 2: 'State',3: 'Zipcode'}, inplace=True)

g = apt.columns.to_series().groupby(apt.dtypes).groups
g

h = dscompszip.columns.to_series().groupby(dscompszip.dtypes).groups
h

#Read zip code data for 5 years from 2012 to 2016

zip2016 = pandas.read_csv('D:\Desktop\Capstone_Greystone\publicdatasets\R11617789_SL860_2016.csv')
zip2016columns = zip2016.columns.get_values()
zip2016columnslist = zip2016columns.tolist()
zip2016columnsdf = pandas.DataFrame(zip2016columnslist)
zip2016columnsdf.to_csv('D:\Desktop\Capstone_Greystone\publicdatasets\selectedcolumns.csv')
zip2016filter = zip2016[['Geo_ZCTA5','SE_T001_001',	'SE_T002_002',	'SE_T004_002',	'SE_T004_003',	'SE_T009_002',	'SE_T009_003',	'SE_T009_004',	'SE_T009_005',	'SE_T013_002',	'SE_T013_003',	'SE_T013_004',	'SE_T013_005',	'SE_T013_006',	'SE_T013_007',	'SE_T013_008',	'SE_T017_001',	'SE_T017_002',	'SE_T017_007',	'SE_T211_001',	'SE_T211_002',	'SE_T211_016',	'SE_T021_001',	'SE_T227_001',	'SE_T028_002',	'SE_T028_003',	'SE_T033_002',	'SE_T033_007',	'SE_T056_002',	'SE_T056_003',	'SE_T056_004',	'SE_T056_005',	'SE_T056_006',	'SE_T056_007',	'SE_T056_008',	'SE_T056_009',	'SE_T056_010',	'SE_T056_011',	'SE_T056_012',	'SE_T056_013',	'SE_T056_014',	'SE_T056_015',	'SE_T056_016',	'SE_T056_017',	'SE_T221_002',	'SE_T221_003',	'SE_T221_004',	'SE_T221_005',	'SE_T221_006',	'SE_T221_007',	'SE_T221_008',	'SE_T221_009',	'SE_T221_010',	'SE_T221_011',	'SE_T221_012',	'SE_T067_002',	'SE_T067_003',	'SE_T083_001',	'SE_T157_001',	'SE_T094_002',	'SE_T094_003',	'SE_T096_002',	'SE_T096_003',	'SE_T096_004',	'SE_T097_002',	'SE_T097_005',	'SE_T097_006',	'SE_T097_007',	'SE_T097_008',	'SE_T097_009',	'SE_T097_010',	'SE_T097_011',	'SE_T097_012',	'SE_T191_002',	'SE_T191_003',	'SE_T191_004',	'SE_T191_005',	'SE_T191_006',	'SE_T191_007',	'SE_T191_008',	'SE_T191_009',	'SE_T191_010',	'SE_T191_011',	'SE_T098_001',	'SE_T100_002',	'SE_T100_003',	'SE_T100_004',	'SE_T100_005',	'SE_T100_006',	'SE_T100_007',	'SE_T100_008',	'SE_T100_009',	'SE_T100_010',	'SE_T102_002',	'SE_T102_003',	'SE_T102_004',	'SE_T102_005',	'SE_T102_006',	'SE_T102_007',	'SE_T102_008',	'SE_T102_009',	'SE_T104_001',	'SE_T108_002',	'SE_T108_008',	'SE_T113_002',	'SE_T113_011',	'SE_T128_002',	'SE_T128_003',	'SE_T128_004',	'SE_T128_005',	'SE_T128_006',	'SE_T128_007',	'SE_T128_008',	'SE_T235_002',	'SE_T235_003',	'SE_T235_004',	'SE_T235_005',	'SE_T235_006',	'SE_T235_007',	'SE_T147_001',	'SE_T182_002',	'SE_T182_003',	'SE_T182_004',	'SE_T182_005',	'SE_T182_006',	'SE_T182_007',	'SE_T199_002',	'SE_T199_003',	'SE_T199_004',	'SE_T199_005',	'SE_T199_006',	'SE_T199_007']]
 
i = zip2016filter.columns.to_series().groupby(zip2016filter.dtypes).groups
i

aptv3 = pandas.merge(aptv1, zip2016filter, how='left', left_on=['Zipcode'],right_on=['Geo_ZCTA5'])
   
#{dtype('int64'): Index(['Zipcode', 'bedroom '], dtype='object'),
# dtype('float64'): Index(['deposit', 'year built', 'year renovated', 'units'], dtype='object'),
# dtype('O'): Index(['Property Name', 'Address', 'City', 'State', 'bathroom', 'rent', 'size',
#        'Monthly Fees', 'One Time Fees', 'Pet Policy', 'Parking', 'Gym',
#        'Kitchen', 'Amenities', 'Features', 'Living Space', 'Lease Info',
#        'Services', 'Property Info', 'Indoor Info', 'Outdoor Info'],
#       dtype='object')}

#Merge dscompszip and aptv1 - only 14 matches on inner join, need fuzzy match
#fuzzy match has given 20 matches
import difflib
#dscompszipapt = pandas.merge(dscompszip, aptv1, how='inner', left_on=['requeststreet'],right_on=['Address'])
#dscompszip[(dscompszip['requeststreet']=='721 Throne Dr')]
dscompszip['Address_final_merge'] = dscompszip['address_final'].apply(lambda x: (difflib.get_close_matches(x, aptv3['Address'],cutoff=0.9)[:1] or [None])[0])
dscompszip['Address_final_merge'].count()

unique_add = dscompszip['Address_final_merge'].drop_duplicates()
#aptv4 = aptv3[aptv3['Address'].isin(unique_add)]
aptv5 = aptv3[~aptv3['Address'].isin(unique_add)]
aptv4 = aptv3[['bathroom','bedroom ','rent','size','units','year built','City','State','Address']]
aptv4.rename(columns={'Address': 'address_final'}, inplace=True)

aptv5 = aptv5.drop(['bathroom','bedroom ','rent','size','units','year built','City','State'], axis=1)
aptv5.rename(columns={'Address': 'address_final', 'Zipcode': 'zipcode_final'}, inplace=True)
dscompszipapt = dscompszip.append(aptv5)
dscompszipapt.Address_final_merge.fillna(dscompszipapt.address_final, inplace=True)
dscompszipapt = pandas.merge(dscompszipapt, aptv4, how='outer', left_on=['Address_final_merge'],right_on=['address_final'])
dscompszipapt = pandas.merge(dscompszipapt, apt_cat, how='left', left_on=['Address_final_merge'],right_on=['Address'])
dscompszipapt.drop(['Address'], axis=1, inplace=True)

i1 = dscompszipapt.columns.to_series().groupby(dscompszipapt.dtypes).groups
i1

#Write aggregated file and miss values to disk - Eugene Deep Search + Deep Comps + Zip 2016 + Aptmts.com
#dscompszipapt.to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\eugene_dscompszipapt.csv')

#Write aggregated file and miss values to disk - Columbia Deep Search + Deep Comps + Zip 2016 + Aptmts.com
#dscompszipapt.to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\columbia_dscompszipapt.csv')

#Write aggregated file and miss values to disk - Columbia Deep Search + Deep Comps + Zip 2016 + Aptmts.com
#dscompszipapt.to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\dayton_dscompszipapt.csv')

#Write aggregated file and miss values to disk - Columbia Deep Search + Deep Comps + Zip 2016 + Aptmts.com
#dscompszipapt.to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\chattanooga_dscompszipapt.csv')






                                                                         
