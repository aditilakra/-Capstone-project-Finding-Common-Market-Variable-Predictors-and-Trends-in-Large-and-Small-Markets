#Import libraries
import pandas
import numpy
import os
import csv

#Use DSCompsZip2016Apt data

#dscompszipapt = pandas.read_csv('D:\Desktop\Capstone_Greystone\Data aggregation\eugene_dscompszipapt.csv')
#dscompszipapt = pandas.read_csv('D:\Desktop\Capstone_Greystone\Data aggregation\columbia_dscompszipapt.csv')
#dscompszipapt = pandas.read_csv('D:\Desktop\Capstone_Greystone\Data aggregation\dayton_dscompszipapt.csv')
#dscompszipapt = pandas.read_csv('D:\Desktop\Capstone_Greystone\Data aggregation\chattanooga_dscompszipapt.csv')

#dscompszipapt.drop('Unnamed: 0', axis=1, inplace=True)

#dscompszipapt = pandas.read_csv('D:\Desktop\Capstone_Greystone\Data aggregation\columbia_dscompszipapt.csv')
#dscompszipapt.drop('Unnamed: 0', axis=1, inplace=True)

#dscompszipapt = pandas.read_csv('D:\Desktop\Capstone_Greystone\Data aggregation\dayton_dscompszipapt.csv')
#dscompszipapt.drop('Unnamed: 0', axis=1, inplace=True)

#dscompszipapt = pandas.read_csv('D:\Desktop\Capstone_Greystone\Data aggregation\chattanooga_dscompszipapt.csv')
#dscompszipapt.drop('Unnamed: 0', axis=1, inplace=True)

#Gather multifamily properties for each market

dscompszipaptv2 = dscompszipapt[(dscompszipapt.useCodenew=="MultiFamily5Plus") | (dscompszipapt.units >= 5)]

#Prepare Data for each market
#bathroom bedroom rent size units year built City State - apartments.com
dscompszipaptfinal=dscompszipaptv2.copy()
dscompszipaptfinal['bedrooms_final'] = dscompszipaptfinal['bedrooms_x']
dscompszipaptfinal['bedrooms_final'].fillna(dscompszipaptfinal['bedrooms_y'], inplace=True)
dscompszipaptfinal['bedrooms_final'].fillna(dscompszipaptfinal['bedroom '], inplace=True)
dscompszipaptfinal.drop(['bedrooms_x','bedrooms_y','bedroom '], axis=1, inplace=True)

j = dscompszipaptfinal.columns.to_series().groupby(dscompszipaptfinal.dtypes).groups
j

dscompszipaptfinal['bathrooms_final'] = dscompszipaptfinal['bathrooms_x']
dscompszipaptfinal['bathrooms_final'].fillna(dscompszipaptfinal['bathrooms_y'], inplace=True)
dscompszipaptfinal['bathrooms_final'].fillna(dscompszipaptfinal['bathroom'], inplace=True)
dscompszipaptfinal.drop(['bathrooms_x','bathrooms_y','bathroom'], axis=1, inplace=True)

dscompszipaptfinal['sqft_final'] = dscompszipaptfinal['finishedSqFt']
dscompszipaptfinal['sqft_final'].fillna(dscompszipaptfinal['sqft'], inplace=True)
dscompszipaptfinal['sqft_final'].fillna(dscompszipaptfinal['size'],inplace=True)
dscompszipaptfinal.drop(['finishedSqFt','sqft','size'], axis=1, inplace=True)

dscompszipaptfinal['yearbuilt_final'] = dscompszipaptfinal['yearbuilt']
dscompszipaptfinal['yearbuilt_final'].fillna(dscompszipaptfinal['year built'], inplace=True)
dscompszipaptfinal.drop(['yearbuilt','year built'], axis=1, inplace=True)

dscompszipaptfinal['units_final'] = dscompszipaptfinal['count']
dscompszipaptfinal['units_final'].fillna(dscompszipaptfinal['units'], inplace=True)
dscompszipaptfinal.loc[dscompszipaptfinal['units_final'] < 5,'units_final'] = 5
dscompszipaptfinal.drop(['count','units'], axis=1, inplace=True)

dscompszipaptfinal['zestimate_amount_final'] = dscompszipaptfinal['zestimate_amount']
dscompszipaptfinal['lastSoldPrice_final'] = dscompszipaptfinal['lastSoldPrice']
dscompszipaptfinal['rent_final'] = dscompszipaptfinal['rent']
dscompszipaptfinal.drop(['zestimate_amount','lastSoldPrice','rent'], axis=1, inplace=True)

dscompszipaptfinal['city_final'] = dscompszipaptfinal['city_x']
dscompszipaptfinal['city_final'].fillna(dscompszipaptfinal['city_y'], inplace=True)
dscompszipaptfinal['city_final'].fillna(dscompszipaptfinal['City'], inplace=True)
dscompszipaptfinal['city_final']="Newyork"


dscompszipaptfinal['state_final'] = dscompszipaptfinal['state_x']
dscompszipaptfinal['state_final'].fillna(dscompszipaptfinal['state_y'], inplace=True)
dscompszipaptfinal['state_final'].fillna(dscompszipaptfinal['State'], inplace=True)

 
dscompszipaptfinal.rename(columns={'address_final_x': 'address_final'}, inplace=True)
dscompszipaptfinal.drop(['Address_final_merge','Geo_ZCTA5','address','city_x','city_y','lastSoldDate','region','region_id','region_type','region_zindexValue','requeststreet','state_x','state_y','zestimate_high','zestimate_lastupdated','zestimate_low','zestimate_percentile','zestimate_valueChange','zipcode_x','zipcode_y','zpid','zpid','City','State','address_final_y'], axis=1,inplace=True)
dscompszipaptfinal.loc[dscompszipaptfinal['useCodenew'] == 'MultiFamily5Plus','source_final'] = 'Zillow.com'
dscompszipaptfinal.loc[dscompszipaptfinal['useCodenew'] != 'MultiFamily5Plus','source_final'] = 'Apartments.com'

#dscompszipaptfinal.to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\eugene_dscompszipaptfinal.csv')
#dscompszipaptfinal.to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\columbia_dscompszipaptfinal.csv')
#dscompszipaptfinal.to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\dayton_dscompszipaptfinal.csv')
#dscompszipaptfinal.to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\chattanooga_dscompszipaptfinal.csv')

