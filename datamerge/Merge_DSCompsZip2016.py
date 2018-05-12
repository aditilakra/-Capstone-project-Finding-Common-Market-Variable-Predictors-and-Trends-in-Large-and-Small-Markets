#Import libraries
import pandas
import numpy
import os
import csv

#Read input files- Eugene
#dsp1 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/eugene/getdeepsearchpart1.xlsx')
#dsp2 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/eugene/getdeepsearchpart2.xlsx')
#comps = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/eugene/getdeepcomps.xlsx')

#Read input files- Columbia
#dsp1 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/columbia/getdeepsearchpart1.xlsx')
#dsp2 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/columbia/getdeepsearchpart2.xlsx')
#comps = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/columbia/getdeepcomps.xlsx')

#Read input files- Dayton
#dsp1 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/dayton/getdeepsearchpart1.xlsx')
#dsp2 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/dayton/getdeepsearchpart2.xlsx')
#comps = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/dayton/getdeepcomps.xlsx')

#Read input files- Chattanooga
#dsp1 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/chattanooga/getdeepsearchpart1.xlsx')
#dsp2 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/chattanooga/getdeepsearchpart2.xlsx')
#comps = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/chattanooga/getdeepcomps.xlsx')

#Read input files- Austin
#dsp1 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/Austin/getdeepsearchpart1.xlsx')
#dsp2 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/Austin/getdeepsearchpart2.xlsx')
#comps = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/Austin/getdeepcomps.xlsx')

#Read input files- Chicago
#dsp1 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/Chicago/getdeepsearchpart1.xlsx')
#dsp2 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/Chicago/getdeepsearchpart2.xlsx')
#comps = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/Chicago/getdeepcomps.xlsx')

#Read input files- Newyork
#dsp1 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/Newyork/getdeepsearchpart1.xlsx')
#dsp2 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/Newyork/getdeepsearchpart2.xlsx')
#comps = pandas.read_excel('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Zillow inputs for code 1/Newyork/getdeepcomps.xlsx')



#Read input files- Orlando
#dsp1 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/zillowscraper/orlando/getdeepsearchpart1_orlando.xlsx')
#dsp2 = pandas.read_excel('//Users/aditilakra/Desktop/capstone/zillowscraper/orlando/getdeepsearchpart2_orlando.xlsx')
#comps = pandas.read_excel('/Users/aditilakra/Desktop/capstone/zillowscraper/orlando/getdeepcomps_orlando.xlsx')



#Read input files- Los Angeles
dsp1 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/zillowscraper/losangeles/getdeepsearchpart1_la.xlsx')
dsp2 = pandas.read_excel('/Users/aditilakra/Desktop/capstone/zillowscraper/losangeles/getdeepsearchpart2_la.xlsx')
comps = pandas.read_excel('/Users/aditilakra/Desktop/capstone/zillowscraper/losangeles/getdeepcomps_la.xlsx')





#print the column names
print(dsp2.columns)

#Roll up getdeepsearch2 to compute count of properties - stored in ctab1
ctab = pandas.crosstab(index=dsp2["requeststreet"],  columns=dsp2["useCode"])
ctab['useCodenew'] = ctab.idxmax(axis=1)
ctab['Index1'] = ctab.index
ctab.rename(columns = {'Index1':'requeststreet'}, inplace = True)
dsp2 = pandas.merge(dsp2, ctab[['requeststreet','useCodenew']], how='left', on=['requeststreet'])
dsp2_v1 = dsp2
#.query('useCodenew=="MultiFamily5Plus"')
ctab1 = pandas.crosstab(index=dsp2_v1["requeststreet"],  columns="count")
ctab1['Index1'] = ctab1.index
ctab1.rename(columns = {'Index1':'requeststreet'}, inplace = True)
dsp2_v2 = dsp2_v1[['requeststreet','state','city','zipcode','useCodenew']]
dsp2_v3 = dsp2_v2.drop_duplicates()
dsp2_v3 = pandas.merge(dsp2_v3, ctab1, how='left', on=['requeststreet'])

#Roll up getdeepsearch1 to average values of property characteristics
dsp1 = pandas.merge(dsp1, ctab[['requeststreet','useCodenew']], how='left', on=['requeststreet'])
dsp1_v1 = dsp1
#.query('useCodenew=="MultiFamily5Plus"')
dsp1_v2 = dsp1_v1.groupby('requeststreet')['finishedSqFt','bathrooms','bedrooms'].mean()
dsp1_v2['Index1'] = dsp1_v2.index
dsp1_v2.rename(columns = {'Index1':'requeststreet'}, inplace = True)

dsp1_v3=dsp1_v1.sort_values("yearbuilt").groupby("requeststreet", as_index=False).first()
dsp1_v3=dsp1_v3[['yearbuilt','requeststreet']]

#Left Join ctab1 with dsp1_v2 & dsp1_v3
#final_ds=dsp2_v1.sort_values("street").groupby("requeststreet", as_index=False).first()
#final_ds=final_ds[['requeststreet','state','city','zipcode','useCodenew']]

final_ds = pandas.merge(dsp2_v3, dsp1_v2, how='left', on=['requeststreet'])
final_ds1 = pandas.merge(final_ds, dsp1_v3, how='left', on=['requeststreet'])

final_ds2 = final_ds1.query('useCodenew=="MultiFamily5Plus"')
#Final files 
#final_ds1 - contains all  property types unit count, query request street, property char
#final_ds2 - contains multifamily5+ unit count, query request street, property char
 
#Merge deepsearch outputs with deepcomps outputs

ds_comps = pandas.merge(final_ds1, comps, how='outer', left_on=['requeststreet'],right_on=['address'])

#Merge deepsearch + deepcomps outputs with zipcode data

#Read zip code data for 5 years from 2012 to 2016

zip2016 = pandas.read_csv('/Users/aditilakra/Desktop/capstone/ACSFiles/R11617789_SL860_2016.csv')
zip2016columns = zip2016.columns.get_values()
zip2016columnslist = zip2016columns.tolist()
zip2016columnsdf = pandas.DataFrame(zip2016columnslist)
#zip2016columnsdf.to_csv('D:\Desktop\Capstone_Greystone\publicdatasets\selectedcolumns.csv')
zip2016filter = zip2016[['Geo_ZCTA5','SE_T001_001',	'SE_T002_002',	'SE_T004_002',	'SE_T004_003',	'SE_T009_002',	'SE_T009_003',	'SE_T009_004',	'SE_T009_005',	'SE_T013_002',	'SE_T013_003',	'SE_T013_004',	'SE_T013_005',	'SE_T013_006',	'SE_T013_007',	'SE_T013_008',	'SE_T017_001',	'SE_T017_002',	'SE_T017_007',	'SE_T211_001',	'SE_T211_002',	'SE_T211_016',	'SE_T021_001',	'SE_T227_001',	'SE_T028_002',	'SE_T028_003',	'SE_T033_002',	'SE_T033_007',	'SE_T056_002',	'SE_T056_003',	'SE_T056_004',	'SE_T056_005',	'SE_T056_006',	'SE_T056_007',	'SE_T056_008',	'SE_T056_009',	'SE_T056_010',	'SE_T056_011',	'SE_T056_012',	'SE_T056_013',	'SE_T056_014',	'SE_T056_015',	'SE_T056_016',	'SE_T056_017',	'SE_T221_002',	'SE_T221_003',	'SE_T221_004',	'SE_T221_005',	'SE_T221_006',	'SE_T221_007',	'SE_T221_008',	'SE_T221_009',	'SE_T221_010',	'SE_T221_011',	'SE_T221_012',	'SE_T067_002',	'SE_T067_003',	'SE_T083_001',	'SE_T157_001',	'SE_T094_002',	'SE_T094_003',	'SE_T096_002',	'SE_T096_003',	'SE_T096_004',	'SE_T097_002',	'SE_T097_005',	'SE_T097_006',	'SE_T097_007',	'SE_T097_008',	'SE_T097_009',	'SE_T097_010',	'SE_T097_011',	'SE_T097_012',	'SE_T191_002',	'SE_T191_003',	'SE_T191_004',	'SE_T191_005',	'SE_T191_006',	'SE_T191_007',	'SE_T191_008',	'SE_T191_009',	'SE_T191_010',	'SE_T191_011',	'SE_T098_001',	'SE_T100_002',	'SE_T100_003',	'SE_T100_004',	'SE_T100_005',	'SE_T100_006',	'SE_T100_007',	'SE_T100_008',	'SE_T100_009',	'SE_T100_010',	'SE_T102_002',	'SE_T102_003',	'SE_T102_004',	'SE_T102_005',	'SE_T102_006',	'SE_T102_007',	'SE_T102_008',	'SE_T102_009',	'SE_T104_001',	'SE_T108_002',	'SE_T108_008',	'SE_T113_002',	'SE_T113_011',	'SE_T128_002',	'SE_T128_003',	'SE_T128_004',	'SE_T128_005',	'SE_T128_006',	'SE_T128_007',	'SE_T128_008',	'SE_T235_002',	'SE_T235_003',	'SE_T235_004',	'SE_T235_005',	'SE_T235_006',	'SE_T235_007',	'SE_T147_001',	'SE_T182_002',	'SE_T182_003',	'SE_T182_004',	'SE_T182_005',	'SE_T182_006',	'SE_T182_007',	'SE_T199_002',	'SE_T199_003',	'SE_T199_004',	'SE_T199_005',	'SE_T199_006',	'SE_T199_007']]

ds_comps['zipcode_final'] = ds_comps['zipcode_x']
ds_comps.zipcode_final.fillna(ds_comps.zipcode_y, inplace=True)

ds_comps['address_final'] = ds_comps['requeststreet']
ds_comps.address_final.fillna(ds_comps.address, inplace=True)

ds_comps_zip = pandas.merge(ds_comps, zip2016filter, how='left', left_on=['zipcode_final'],right_on=['Geo_ZCTA5'])

#Write aggregated file and miss values to disk - Eugene Deep Search + Deep Comps + Zip 2016
#ds_comps_zip.to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/eugene_dscompszip.csv')

#(ds_comps_zip.isnull().sum()/len(ds_comps_zip.index)*100).to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/eugene_dscompszip_miss.csv')

#Write aggregated file and miss values to disk - Columbia Deep Search + Deep Comps + Zip 2016
#ds_comps_zip.to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/columbia_dscompszip.csv')

#(ds_comps_zip.isnull().sum()/len(ds_comps_zip.index)*100).to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/columbia_dscompszip_miss.csv')

#Write aggregated file and miss values to disk - dayton Deep Search + Deep Comps + Zip 2016
#ds_comps_zip.to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/dayton_dscompszip.csv')

#(ds_comps_zip.isnull().sum()/len(ds_comps_zip.index)*100).to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/dayton_dscompszip_miss.csv')

#Write aggregated file and miss values to disk - chattanooga Deep Search + Deep Comps + Zip 2016
#ds_comps_zip.to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/chattanooga_dscompszip.csv')

#(ds_comps_zip.isnull().sum()/len(ds_comps_zip.index)*100).to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/chattanooga_dscompszip_miss.csv')

#Write aggregated file and miss values to disk - Austin Deep Search + Deep Comps + Zip 2016
#ds_comps_zip.to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/austin_dscompszip.csv')

#(ds_comps_zip.isnull().sum()/len(ds_comps_zip.index)*100).to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/austin_dscompszip_miss.csv')

#Write aggregated file and miss values to disk - Chicago Deep Search + Deep Comps + Zip 2016
#ds_comps_zip.to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/chicago_dscompszip.csv')

#(ds_comps_zip.isnull().sum()/len(ds_comps_zip.index)*100).to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/chicago_dscompszip_miss.csv')

##Write aggregated file and miss values to disk - Newyork Deep Search + Deep Comps + Zip 2016
#ds_comps_zip.to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/newyork_dscompszip.csv')

#(ds_comps_zip.isnull().sum()/len(ds_comps_zip.index)*100).to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/newyork_dscompszip_miss.csv')


ds_comps_zip.to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/la_dscompszip.csv')

(ds_comps_zip.isnull().sum()/len(ds_comps_zip.index)*100).to_csv('/Users/aditilakra/Desktop/capstone/Capstone_combined data/Data aggregation/la_dscompszip_miss.csv')
