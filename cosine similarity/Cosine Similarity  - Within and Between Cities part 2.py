#Import libraries
import pandas
import numpy
import os
import csv
import difflib
from collections import Counter

#dscompszipapt1 = pandas.read_csv('D:\Desktop\Capstone_Greystone\Data aggregation\eugene_dscompszipaptfinal.csv')
#dscompszipapt1.drop('Unnamed: 0', axis=1, inplace=True)

#dscompszipapt2 = pandas.read_csv('D:\Desktop\Capstone_Greystone\Data aggregation\columbia_dscompszipaptfinal.csv')
#dscompszipapt2.drop('Unnamed: 0', axis=1, inplace=True)
dscompszipapt1 = dscompszipaptfinal.copy()
dscompszipapt2 = dscompszipaptfinal.copy()
dscompszipapt3 = dscompszipaptfinal.copy()
dscompszipapt4 = dscompszipaptfinal.copy()
dscompszipapt5 = dscompszipaptfinal.copy()
dscompszipapt6 = dscompszipaptfinal.copy()
dscompszipapt7 = dscompszipaptfinal.copy()

#dscompszipapt3 = pandas.read_csv('D:\Desktop\Capstone_Greystone\Data aggregation\dayton_dscompszipaptfinal.csv')
#dscompszipapt4 = pandas.read_csv('D:\Desktop\Capstone_Greystone\Data aggregation\chattanooga_dscompszipaptfinal.csv')

#Aggregate all markets and filter for records with rent

dscompszipapt4append = dscompszipapt1.append(dscompszipapt2)
dscompszipapt4append = dscompszipapt4append.append(dscompszipapt3)
dscompszipapt4append = dscompszipapt4append.append(dscompszipapt4)
dscompszipapt4append = dscompszipapt4append.append(dscompszipapt5)
dscompszipapt4append = dscompszipapt4append.append(dscompszipapt6)
dscompszipapt4append = dscompszipapt4append.append(dscompszipapt7)

dscompszipapt4append
dscompszipapt4append1 = dscompszipapt4append.loc[dscompszipapt4append.rent_final.notnull(),:]

#dscompszipapt4append1.groupby(["state_final", "city_final"]).size()
 
#Export data for visualization
dscompszipapt4append1.to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\dscompszipapt4append1.csv')

k = dscompszipapt4append1.columns.to_series().groupby(dscompszipapt4append1.dtypes).groups
k

#Fix Property Age variable
dscompszipapt4append1['age_yrs_final'] = 2018 - dscompszipapt4append1['yearbuilt_final'] 
dscompszipapt4append1[['age_yrs_final','yearbuilt_final']].head()
dscompszipapt4append1.drop(['yearbuilt_final'],axis=1,inplace=True)

#Create binary variables from categorical features
# 'Monthly Fees','One Time Fees', 'Pet Policy', 'Parking', 'Gym', 'Kitchen', 'Amenities',
#'Features', 'Living Space', 'Lease Info', 'Services', 'Indoor Info','Outdoor Info',
dscompszipapt4append1 = dscompszipapt4append1.reset_index()
dscompszipapt4append1.drop(['index'],axis=1,inplace=True)

#Variables discarded - Pet Policy

#Function to split string variables - Monthly Fees, One Time Fees
def cat1(str1,str2,str3):
    global catvar
    fundummy = dscompszipapt4append1[str1]
    splitvar = fundummy.str.split(',',expand=True)
    splitvar = pandas.DataFrame(splitvar)
    splitvar_mod = pandas.DataFrame(index=splitvar.index)
    splitvar_mod[str2] = ""
    for i in range(0,len(splitvar)):
        for j in range(0,len(splitvar.columns)):
            if splitvar.iloc[i,j] == str3:
                assn = splitvar.iloc[i,j+1]
                splitvar_mod.ix[i,str2] = assn                
    catvar =  pandas.merge(catvar, splitvar_mod, left_index=True, right_index=True) 

#Function to store extracted string variables - Monthly Fees, One Time Fees
def cat2(str4):
    global catvar
    catvar[str4] = catvar[str4].str.replace('$', '')
    try:
        catvar[str4] = pandas.to_numeric(catvar[str4])
    except ValueError:
        temp = catvar[str4].str.split('-',expand=True)
        temp[1] = pandas.to_numeric(temp[1])
        temp[0] = pandas.to_numeric(temp[0])
        temp[2] = temp.mean(axis=1)
        catvar[str4] = temp[2]

#Function to store and extract string variables - Parking
def cat3(str5,str6,str7):
    global catvar
    fundummy = dscompszipapt4append1[str5]
    splitvar = pandas.DataFrame(fundummy)
    splitvar["lower"]=splitvar[str5].str.lower()
    splitvar['Match'] = splitvar["lower"].str.contains(str7)       
    splitvar_mod = pandas.DataFrame(index=splitvar.index)
    splitvar_mod[str6] = ""
    for i in range(0,len(splitvar)):
        if splitvar.iloc[i,2] == True:
            splitvar_mod.ix[i,str6] = 1
        elif pandas.notnull(splitvar.iloc[i,0]):
            splitvar_mod.ix[i,str6] = 0
        else:
            splitvar_mod.ix[i,str6] = numpy.nan
    splitvar_mod[str6]=splitvar_mod[str6].astype(float)
    catvar =  pandas.merge(catvar, splitvar_mod, left_index=True, right_index=True) 

##Variable extracted - Monthly Fees

#Unassigned Surface Lot Parking, Assigned Surface Lot Parking, Assigned Covered Parking, Assigned Garage Parking, Unassigned Garage Parking
#Assigned Other Parking, Unassigned Other Parking,  Unassigned Covered Parking, Other Rent, Storage Fee, Dog Rent, Cat Rent

#Create empty data frame to store transformed string variables
catvar = pandas.DataFrame(index=dscompszipapt4append1.index)

cat1(str1='Monthly Fees',str2="Unassigned_Surface_Lot_Parking",str3='Unassigned Surface Lot Parking')
cat1(str1='Monthly Fees',str2="Assigned_Surface_Lot_Parking",str3='Assigned Surface Lot Parking')
cat1(str1='Monthly Fees',str2="Assigned_Covered_Parking",str3='Assigned Covered Parking')
cat1(str1='Monthly Fees',str2="Assigned_Garage_Parking",str3='Assigned Garage Parking')
cat1(str1='Monthly Fees',str2="Unassigned_Garage_Parking",str3='Unassigned Garage Parking')
cat1(str1='Monthly Fees',str2="Assigned_Other_Parking",str3='Assigned Other Parking')
cat1(str1='Monthly Fees',str2="Unassigned_Other_Parking",str3='Unassigned Other Parking')
cat1(str1='Monthly Fees',str2="Unassigned_Covered_Parking",str3='Unassigned Covered Parking')
cat1(str1='Monthly Fees',str2="Other_Rent",str3='Other Rent')
cat1(str1='Monthly Fees',str2="Storage_Fee",str3='Storage Fee')
cat1(str1='Monthly Fees',str2="Dog_Rent",str3='Dog Rent')
cat1(str1='Monthly Fees',str2="Cat_Rent",str3='Cat Rent')

#k = catvar.columns.to_series().groupby(catvar.dtypes).groups
#k

cat2('Unassigned_Surface_Lot_Parking')        
cat2('Assigned_Surface_Lot_Parking')        
cat2('Assigned_Covered_Parking')        
cat2('Assigned_Garage_Parking')        
cat2('Unassigned_Garage_Parking')        
cat2('Assigned_Other_Parking')        
cat2('Unassigned_Other_Parking')        
cat2('Unassigned_Covered_Parking')        
cat2('Other_Rent')        
cat2('Storage_Fee')        
cat2('Dog_Rent')        
cat2('Cat_Rent')        

#(catvar.isnull().sum()/len(catvar.index)*100)

catvar['Rent_parking'] = catvar[['Unassigned_Surface_Lot_Parking', 'Assigned_Surface_Lot_Parking',
        'Assigned_Covered_Parking', 'Assigned_Garage_Parking',
        'Unassigned_Garage_Parking', 'Unassigned_Other_Parking',
        'Assigned_Other_Parking', 'Unassigned_Covered_Parking']].dropna(thresh=1).mean(axis=1)

dscompszipapt4append1 = pandas.merge(dscompszipapt4append1, catvar[['Rent_parking','Cat_Rent','Dog_Rent']], left_index=True, right_index=True)      

##Variable extracted - One Time Fees

#Amenity Fee,  Cat Fee, Cat Deposit, Admin Fee, Dog Fee, Dog Deposit, Application Fee 

#Create empty data frame to store transformed string variables
catvar = pandas.DataFrame(index=dscompszipapt4append1.index)

cat1(str1='One Time Fees',str2="Amenity_Fee",str3='Amenity Fee')
cat1(str1='One Time Fees',str2="Cat_Fee",str3='Cat Fee')
cat1(str1='One Time Fees',str2="Cat_Deposit",str3='Cat Deposit')
cat1(str1='One Time Fees',str2="Admin_Fee",str3='Admin Fee')
cat1(str1='One Time Fees',str2="Dog_Fee",str3='Dog Fee')
cat1(str1='One Time Fees',str2="Dog_Deposit",str3='Dog Deposit')
cat1(str1='One Time Fees',str2="Application_Fee",str3='Application Fee')

cat2('Amenity_Fee')        
cat2('Cat_Fee')        
cat2('Cat_Deposit')        
cat2('Admin_Fee')        
cat2('Dog_Fee')        
cat2('Dog_Deposit')        
cat2('Application_Fee')        
 
#(catvar.isnull().sum()/len(catvar.index)*100)

catvar['Fee_Application'] = catvar[['Amenity_Fee','Admin_Fee','Application_Fee']].dropna(thresh=1).sum(axis=1)
catvar['Deposit_Cat'] = catvar[['Cat_Fee','Cat_Deposit']].dropna(thresh=1).mean(axis=1)
catvar['Deposit_Dog'] = catvar[['Dog_Fee','Dog_Deposit']].dropna(thresh=1).mean(axis=1)

dscompszipapt4append1 = pandas.merge(dscompszipapt4append1, catvar[['Fee_Application','Deposit_Cat','Deposit_Dog']], left_index=True, right_index=True)      

##Variable not extracted - Pet Policy; since the text is too difficult to decipher

##Variable extracted - Parking

catvar = pandas.DataFrame(index=dscompszipapt4append1.index)

cat3('Parking',"Surface_Lot",'surface lot')
cat3('Parking',"Covered",'covered')
cat3('Parking',"Garage",'garage')
cat3('Parking',"Street",'street')
cat3('Parking',"Multiple_parking_spaces",'spaces')
cat3('Parking',"Assigned_Parking",'assigned parking')

dscompszipapt4append1 = pandas.merge(dscompszipapt4append1, catvar, left_index=True, right_index=True)      

#(dscompszipapt4append1.isnull().sum()/len(dscompszipapt4append1.index)*100)

##Variable extracted - Gym

test = dscompszipapt4append1['Gym']
test1 = test.str.split(',',expand=True)
test2 = list(test1.values.flatten())
test3 = [x for x in test2 if x != None]
test4 = [str(x) for x in test3]
test5 = [x.strip() for x in test4]
Counter(test5).most_common()

#('Fitness Center', 851),
#('Pool', 783),
#('nan', 296),
#('Cardio Machines', 278),
#('Free Weights', 269),
#('Weight Machines', 228),
#('Playground', 223),
#('Bike Storage', 206),
#('Spa', 145),
#('Tennis Court', 116),
#('Gameroom', 114),
#('Walking/Biking Trails', 104),
#('Fitness Programs', 98),
#('Basketball Court', 97),
#('Media Center/Movie Theatre', 94),
#('Volleyball Court', 86),
#('Sauna', 55),
#('Health Club Facility', 45),
#('Racquetball Court', 19),
#('Sport Court', 19),
#('Gaming Stations', 18),
#('Putting Greens', 8)

catvar = pandas.DataFrame(index=dscompszipapt4append1.index)
         
cat3('Gym',"Fitness_Center",'fitness center')
cat3('Gym',"Pool",'pool')
cat3('Gym',"Cardio_Machines",'cardio machines')
cat3('Gym',"Free_Weights",'free weights')
cat3('Gym',"Weight_Machines",'weight machines')
cat3('Gym',"Playground",'playground')
cat3('Gym',"Bike_Storage",'bike storage')
cat3('Gym',"Spa",'spa')
cat3('Gym',"Tennis_Court",'tennis court')
cat3('Gym',"Gameroom",'gameroom')
cat3('Gym',"WalkingBiking_Trails",'walking/biking trails')
cat3('Gym',"Fitness_Programs",'fitness programs')
cat3('Gym',"Basketball_Court",'basketball court')
cat3('Gym',"MediaCenter_MovieTheatre",'media center/movie theatre')
cat3('Gym',"Volleyball_Court",'volleyball court')

dscompszipapt4append1 = pandas.merge(dscompszipapt4append1, catvar, left_index=True, right_index=True)      

#pandas.crosstab(index=catvar["MediaCenter_MovieTheatre"],  columns="count")
#(dscompszipapt4append1.isnull().sum()/len(dscompszipapt4append1.index)*100)

##Variable extracted - Kitchen

test = dscompszipapt4append1['Kitchen']
test1 = test.str.split(',',expand=True)
test2 = list(test1.values.flatten())
test3 = [x for x in test2 if x != None]
test4 = [str(x) for x in test3]
test5 = [x.strip() for x in test4]
Counter(test5).most_common()

#('Dishwasher', 926),
# ('Range', 854),
# ('Refrigerator', 824),
# ('Kitchen', 722),
# ('Microwave', 655),
# ('Disposal', 638),
# ('Oven', 580),
# ('Stainless Steel Appliances', 333),
# ('Ice Maker', 291),
# ('Granite Countertops', 281),
# ('Freezer', 255),
# ('Pantry', 221),
# ('nan', 200),
# ('Eat-in Kitchen', 193),
# ('Island Kitchen', 129),
# ('Breakfast Nook', 44),
# ('Instant Hot Water', 40),
# ('Warming Drawer', 7),
# ('Coffee System', 4)
 
catvar = pandas.DataFrame(index=dscompszipapt4append1.index)
         
cat3('Kitchen',"Dishwasher",'dishwasher')
cat3('Kitchen',"Range",'range')
cat3('Kitchen',"Refrigerator",'refrigerator')
cat3('Kitchen',"Kitchen",'kitchen')
cat3('Kitchen',"Microwave",'microwave')
cat3('Kitchen',"Disposal",'disposal')
cat3('Kitchen',"Oven",'oven')
cat3('Kitchen',"Stainless_Steel_Appliances",'stainless steel appliances')
cat3('Kitchen',"Ice_Maker",'ice_maker')
cat3('Kitchen',"Granite_Countertops",'granite countertops')
cat3('Kitchen',"Freezer",'freezer')
cat3('Kitchen',"Pantry",'pantry')
cat3('Kitchen',"Eatin_Kitchen",'eat-in kitchen')
cat3('Kitchen',"Island_Kitchen",'island kitchen')

dscompszipapt4append1 = pandas.merge(dscompszipapt4append1, catvar, left_index=True, right_index=True)      

#pandas.crosstab(index=catvar["Dishwasher"],  columns="count")
#(dscompszipapt4append1.isnull().sum()/len(dscompszipapt4append1.index)*100)

##Variable extracted - Amenities

test = dscompszipapt4append1['Amenities']
test1 = test.str.split(',',expand=True)
test2 = list(test1.values.flatten())
test3 = [x for x in test2 if x != None]
test4 = [str(x) for x in test3]
test5 = [x.strip() for x in test4]
test6 = Counter(test5).most_common()
test7 = pandas.DataFrame(test6)
test7.to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/Amenities.csv')
    
#nan	429
#Dishwasher	172
#Hardwood Floors	127
#Package Receiving	124
#Cable Ready	114
#Large Closets	100
#Refrigerator	97
#Microwave	90
#BBQ/Picnic Area	86
#Stainless Steel Appliances	85
#Disposal	83
#Air Conditioner	82
#Granite Countertops	80
#Ceiling Fan	75
#Window Coverings	71
#Patio/Balcony	71
#Extra Storage	70
 
catvar = pandas.DataFrame(index=dscompszipapt4append1.index)
         
cat3('Kitchen',"Dishwasher",'dishwasher')
cat3('Kitchen',"Range",'range')
cat3('Kitchen',"Refrigerator",'refrigerator')
cat3('Kitchen',"Kitchen",'kitchen')
cat3('Kitchen',"Microwave",'microwave')
cat3('Kitchen',"Disposal",'disposal')
cat3('Kitchen',"Oven",'oven')
cat3('Kitchen',"Stainless_Steel_Appliances",'stainless steel appliances')
cat3('Kitchen',"Ice_Maker",'ice_maker')
cat3('Kitchen',"Granite_Countertops",'granite countertops')
cat3('Kitchen',"Freezer",'freezer')
cat3('Kitchen',"Pantry",'pantry')
cat3('Kitchen',"Eatin_Kitchen",'eat-in kitchen')
cat3('Kitchen',"Island_Kitchen",'island kitchen')

dscompszipapt4append1 = pandas.merge(dscompszipapt4append1, catvar, left_index=True, right_index=True)      

#pandas.crosstab(index=catvar["Dishwasher"],  columns="count")
#(dscompszipapt4append1.isnull().sum()/len(dscompszipapt4append1.index)*100)

#Imputation - Mean based

X = dscompszipapt4append1.drop(['city_final','state_final','useCodenew','zipcode_final','address_final','rent_final','source_final'], axis=1)
Xmiss = (X.isnull().sum()/len(X.index)*100)
#sqft_final         0.293255 %
#age_yrs_final    5.278592 %

#listX = list(X)

k = X.columns.to_series().groupby(X.dtypes).groups
k

X = X.apply(lambda x: x.fillna(x.mean()),axis=0)


#Correlation
corr_matrix = dscompszipapt4append1.corr()
corr_matrix["rent_final"].sort_values(ascending=False).to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\Analysis\correlation_4markets.csv')

from pandas.tools.plotting import scatter_matrix
attributes = ["rent_final","sqft_final","yearbuilt_final","bathrooms_final","bedrooms_final","units_final"]
scatter_matrix(housing[attributes], figsize=(12, 8))

#Run Similarities across markets

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity = cosine_similarity(X)
cosine_similarity_df = pandas.DataFrame(cosine_similarity)
cosine_similarity_df_stack = cosine_similarity_df.stack()
cosine_similarity_stack_df = pandas.DataFrame(cosine_similarity_df_stack)
cosine_similarity_stack_df['Index1'] = cosine_similarity_stack_df.index
cosine_similarity_stack_df['Index1'] = cosine_similarity_stack_df['Index1'].astype(str)
cosine_similarity_stack_df['Index1_Rows'], cosine_similarity_stack_df['Index1_Columns'] = cosine_similarity_stack_df['Index1'].str.split(',', 1).str
cosine_similarity_stack_df['Index1_Rows'] = cosine_similarity_stack_df['Index1_Rows'].str.replace(r"\(","")
cosine_similarity_stack_df['Index1_Columns'] = cosine_similarity_stack_df['Index1_Columns'].str.replace(r"\)","")
cosine_similarity_stack_df = cosine_similarity_stack_df[cosine_similarity_stack_df['Index1_Rows']!="'Index'"]
cosine_similarity_stack_df['Index1_Rows'] = pandas.to_numeric(cosine_similarity_stack_df['Index1_Rows'])
cosine_similarity_stack_df['Index1_Columns'] = pandas.to_numeric(cosine_similarity_stack_df['Index1_Columns'])
cosine_similarity_stack_df = cosine_similarity_stack_df[cosine_similarity_stack_df['Index1_Rows'] != cosine_similarity_stack_df['Index1_Columns']]
cosine_similarity_stack_df = cosine_similarity_stack_df.sort_values(['Index1_Rows', 0], ascending=[True, False])
cosine_similarity_stack_df = cosine_similarity_stack_df.groupby('Index1_Rows').head(30)

Y = dscompszipapt4append1.reset_index(drop=True)
Y['Index1'] = Y.index
Y['Index1'] = Y['Index1'].astype(int)
Y=Y[['Index1','rent_final']]
cosine_similarity_stack_df = pandas.merge(cosine_similarity_stack_df, Y, how='left', left_on=['Index1_Columns'],right_on=['Index1'])
cosine_similarity_stack_df = cosine_similarity_stack_df.drop(['Index1_x','Index1_y'], axis=1)

k1 = cosine_similarity_stack_df.columns.to_series().groupby(cosine_similarity_stack_df.dtypes).groups
k1

cosine_similarity_stack_df['Index1_Columns2'] = cosine_similarity_stack_df.groupby('Index1_Rows').cumcount()
cosine_similarity_stack_df['Index1_Columns2'] = cosine_similarity_stack_df['Index1_Columns2'].astype(str)
cosine_similarity_stack_df['Index1_Rows'] = cosine_similarity_stack_df['Index1_Rows'].astype(str)

cosine_similarity_stack_df.rename(columns={0: 'Cosine_sim'}, inplace=True)

mlinput = pandas.pivot_table(cosine_similarity_stack_df,index=["Index1_Rows"],columns=["Index1_Columns2"],values=["rent_final","Cosine_sim"])

#Temporarily share outputs for the call

#dscompszipapt4append1.to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\dscompszipapt4append1.csv')
#cosine_similarity_stack_df.to_csv('D:\Desktop\Capstone_Greystone\Data aggregation\cosine_similarity_stack_df.csv')

#Split data into train and test (30%)
#numpy.random.seed(42)
#def split_train_test(data, test_ratio):    
#    shuffled_indices = numpy.random.permutation(len(data))
#    test_set_size = int(len(data) * test_ratio)
#    test_indices = shuffled_indices[:test_set_size]
#    train_indices = shuffled_indices[test_set_size:]
#    return data.iloc[train_indices], data.iloc[test_indices]

#train_set, test_set = split_train_test(dscompszipapt4append1, 0.3)
#print(len(train_set), "train +", len(test_set), "test")

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dscompszipapt4append1, test_size=0.3, random_state=42)
