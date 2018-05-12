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
 
#Fix Property Age variable
    dscompszipapt4append1['age_yrs_final'] = 2018 - dscompszipapt4append1['yearbuilt_final'] 
dscompszipapt4append1[['age_yrs_final','yearbuilt_final']].head()
dscompszipapt4append1.drop(['yearbuilt_final'],axis=1,inplace=True)

#Create binary variables from categorical features
# 'Monthly Fees','One Time Fees', 'Pet Policy', 'Parking', 'Gym', 'Kitchen', 'Amenities',
#'Features', 'Living Space', 'Lease Info', 'Services', 'Indoor Info','Outdoor Info',
dscompszipapt4append1 = dscompszipapt4append1.reset_index()
dscompszipapt4append1.drop(['index'],axis=1,inplace=True)

#Variables discarded - Pet Policy, Amenities

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

#Function to store and extract string variables - Parking, Gym, Kitchen, Features, Living Space, Services, Indoor Info, Outdoor Info
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

##Variable extracted - Amenities - Variable ignored as it has same information present in variables Kitchen and Features

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
 
#catvar = pandas.DataFrame(index=dscompszipapt4append1.index)
#cat3('Kitchen',"Dishwasher",'dishwasher')
#dscompszipapt4append1 = pandas.merge(dscompszipapt4append1, catvar, left_index=True, right_index=True)      

#pandas.crosstab(index=catvar["Dishwasher"],  columns="count")
#(dscompszipapt4append1.isnull().sum()/len(dscompszipapt4append1.index)*100)

##Variable extracted - Features

test = dscompszipapt4append1['Features']
test1 = test.str.split(',',expand=True)
test2 = list(test1.values.flatten())
test3 = [x for x in test2 if x != None]
test4 = [str(x) for x in test3]
test5 = [x.strip() for x in test4]
Counter(test5).most_common()
    
#('High Speed Internet Access', 989),
# ('Air Conditioning', 906),
# ('Heating', 866),
# ('Cable Ready', 742),
# ('Tub/Shower', 608),
# ('Smoke Free', 595),
# ('Ceiling Fans', 593),
# ('Washer/Dryer', 477),
# ('Washer/Dryer Hookup', 370),
# ('Storage Units', 353),
# ('Wheelchair Accessible (Rooms)', 327),
# ('Fireplace', 233),
# ('Wi-Fi', 163),
# ('Sprinkler System', 135),
# ('Alarm', 101),
# ('Double Vanities', 92),
# ('Framed Mirrors', 81),
# ('Intercom', 79),
# ('nan', 63),
# ('Handrails', 59),
# ('Trash Compactor', 59),
# ('Satellite TV', 44),
# ('Surround Sound', 10),
# ('Vacuum System', 2) 

catvar = pandas.DataFrame(index=dscompszipapt4append1.index)

cat3('Features',"High_Speed_Internet_Access",'high speed internet access')
cat3('Features',"Air_Conditioning",'air conditioning')
cat3('Features',"Heating",'heating')
cat3('Features',"Cable_Ready",'cable ready')
cat3('Features',"Tub_Shower",'tub/shower')
cat3('Features',"Smoke_Free",'smoke free')
cat3('Features',"Ceiling_Fans",'ceiling fans')
cat3('Features',"Washer_Dryer",'washer/dryer')
cat3('Features',"Storage_Units",'storage units')
cat3('Features',"Wheelchair_Accessible_Rooms",'wheelchair accessible \(rooms\)')
cat3('Features',"Fireplace",'fireplace')
cat3('Features',"Wi_Fi",'wi-fi')
cat3('Features',"Sprinkler_System",'sprinkler system')
cat3('Features',"Alarm",'alarm')
cat3('Features',"Double_Vanities",'double vanities')
cat3('Features',"Framed_Mirrors",'framed mirrors')
cat3('Features',"Intercom",'intercom')

dscompszipapt4append1 = pandas.merge(dscompszipapt4append1, catvar, left_index=True, right_index=True)      

#pandas.crosstab(index=catvar["High_Speed_Internet_Access"],  columns="count")
#(dscompszipapt4append1.isnull().sum()/len(dscompszipapt4append1.index)*100)

##Variable extracted - Living Space

test = dscompszipapt4append1['Living Space']
test1 = test.str.split(',',expand=True)
test2 = list(test1.values.flatten())
test3 = [x for x in test2 if x != None]
test4 = [str(x) for x in test3]
test5 = [x.strip() for x in test4]
Counter(test5).most_common()
    
#[('Walk-In Closets', 721),
# ('Hardwood Floors', 506),
# ('Window Coverings', 462),
# ('Carpet', 461),
# ('Dining Room', 373),
# ('nan', 366),
# ('Views', 360),
# ('Vinyl Flooring', 233),
# ('Tile Floors', 211),
# ('Linen Closet', 181),
# ('Vaulted Ceiling', 153),
# ('Crown Molding', 125),
# ('Double Pane Windows', 112),
# ('Built-In Bookshelves', 96),
# ('Den', 81),
# ('Accent Walls', 74),
# ('Bay Window', 49),
# ('Family Room', 45),
# ('Loft Layout', 43),
# ('Sunroom', 42),
# ('Office', 40),
# ('Furnished', 25),
# ('Skylight', 18),
# ('Recreation Room', 16),
# ('Mother-in-law Unit', 15),
# ('Mud Room', 10),
# ('Wet Bar', 9),
# ('Basement', 5),
# ('Attic', 1),
# ('Workshop', 1)]

catvar = pandas.DataFrame(index=dscompszipapt4append1.index)

cat3('Living Space',"Walk_In_Closets",'walk-in closets')
cat3('Living Space',"Hardwood_Floors",'hardwood floors')
cat3('Living Space',"Window_Coverings",'window coverings')
cat3('Living Space',"Carpet",'carpet')
cat3('Living Space',"Dining_Room",'dining room')
cat3('Living Space',"Views",'views')
cat3('Living Space',"Vinyl_Flooring",'vinyl flooring')
cat3('Living Space',"Tile_Floors",'tile floors')
cat3('Living Space',"Linen_Closet",'linen closet')
cat3('Living Space',"Vaulted_Ceiling",'vaulted ceiling')
cat3('Living Space',"Crown_Molding",'crown molding')
cat3('Living Space',"Double_Pane_Windows",'double pane windows')
cat3('Living Space',"Built_In_Bookshelves",'built-in bookshelves')
cat3('Living Space',"Den",'den')
cat3('Living Space',"Accent_Walls",'accent walls')

dscompszipapt4append1 = pandas.merge(dscompszipapt4append1, catvar, left_index=True, right_index=True)      

#pandas.crosstab(index=catvar["Walk_In_Closets"],  columns="count")
#(dscompszipapt4append1.isnull().sum()/len(dscompszipapt4append1.index)*100)

##Variable extracted - Services

test = dscompszipapt4append1['Services']
test1 = test.str.split(',',expand=True)
test2 = list(test1.values.flatten())
test3 = [x for x in test2 if x != None]
test4 = [str(x) for x in test3]
test5 = [x.strip() for x in test4]
test6 = Counter(test5).most_common()
test7 = pandas.DataFrame(test6)
test7.to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/Services.csv')
    
#('Laundry Facilities', 743),
# ('Maintenance on site', 705),
# ('Property Manager on Site', 568),
# ('Package Service', 417),
# ('Controlled Access', 392),
# ('24 Hour Availability', 336),
# ('Recycling', 282),
# ('Wi-Fi at Pool and Clubhouse', 240),
# ('Online Services', 234),
# ('Pet Play Area', 200),
# ('Planned Social Activities', 197),
# ('Concierge', 166),
# ('Doorman', 155),
# ('nan', 154),
# ('Courtesy Patrol', 153),
# ('Renters Insurance Program', 143),
# ('Dry Cleaning Service', 116),
# ('Laundry Service', 110),
# ('Trash Pickup - Door to Door', 108),
# ('Pet Care', 104),
# ('On-Site Retail', 83),
# ('Pet Washing Station', 82),
# ('Security System', 80),
# ('Furnished Units Available', 77),
# ('Guest Apartment', 71),
# ('Bilingual', 71),
 
catvar = pandas.DataFrame(index=dscompszipapt4append1.index)

cat3('Services',"Laundry_Facilities",'laundry facilities')
cat3('Services',"Maintenance_on_site",'maintenance on site')
cat3('Services',"Property_Manager_on_Site",'property manager on site')
cat3('Services',"Package_Service",'package service')
cat3('Services',"Controlled_Access",'controlled access')
cat3('Services',"24_Hour_Availability",'24 hour availability')
cat3('Services',"Recycling",'recycling')
cat3('Services',"WiFi_at_Pool_and_Clubhouse",'wi-fi at pool and clubhouse')
cat3('Services',"Online_Services",'online services')
cat3('Services',"Pet_Play_Area",'pet play area')
cat3('Services',"Planned_Social_Activities",'planned social activities')
cat3('Services',"Concierge",'concierge')
cat3('Services',"Doorman",'doorman')
cat3('Services',"Courtesy_Patrol",'courtesy patrol')
cat3('Services',"Renters_Insurance_Program",'renters insurance program')
cat3('Services',"Dry_Cleaning_Service",'dry cleaning service')
cat3('Services',"Trash_Pickup_Door_to_Door",'trash pickup - door to door')
cat3('Services',"Pet_Care",'pet care')
cat3('Services',"On_Site_Retail",'on-site retail')
cat3('Services',"Pet_Washing_Station",'pet washing station')
cat3('Services',"Security_System",'security system')

dscompszipapt4append1 = pandas.merge(dscompszipapt4append1, catvar, left_index=True, right_index=True)      

#pandas.crosstab(index=catvar["Laundry_Facilities"],  columns="count")
#(dscompszipapt4append1.isnull().sum()/len(dscompszipapt4append1.index)*100)

##Variable extracted - Indoor Info

test = dscompszipapt4append1['Indoor Info']
test1 = test.str.split(',',expand=True)
test2 = list(test1.values.flatten())
test3 = [x for x in test2 if x != None]
test4 = [str(x) for x in test3]
test5 = [x.strip() for x in test4]
Counter(test5).most_common()

#('Business Center', 550),
# ('Clubhouse', 524),
# ('nan', 399),
# ('Elevator', 358),
# ('Storage Space', 246),
# ('Lounge', 244),
# ('Coffee Bar', 208),
# ('Conference Room', 111),
# ('Multi Use Room', 99),
# ('Disposal Chutes', 77),
# ('Vintage Building', 64),
# ('Breakfast/Coffee Concierge', 58),
# ('Corporate Suites', 45),
# ('Library', 44),
# ('Two Story Lobby', 21),
# ('Tanning Salon', 19)
    
catvar = pandas.DataFrame(index=dscompszipapt4append1.index)

cat3('Indoor Info',"Business_Center",'business center')
cat3('Indoor Info',"Clubhouse",'clubhouse')
cat3('Indoor Info',"Elevator",'elevator')
cat3('Indoor Info',"Storage_Space",'storage space')
cat3('Indoor Info',"Lounge",'lounge')
cat3('Indoor Info',"Coffee_Bar",'coffee bar')
cat3('Indoor Info',"Conference_Room",'conference room')
cat3('Indoor Info',"Multi_Use_Room",'multi use room')
cat3('Indoor Info',"Disposal_Chutes",'disposal chutes')

dscompszipapt4append1 = pandas.merge(dscompszipapt4append1, catvar, left_index=True, right_index=True)      

#pandas.crosstab(index=catvar["Business_Center"],  columns="count")
#(dscompszipapt4append1.isnull().sum()/len(dscompszipapt4append1.index)*100)

##Variable extracted - Outdoor Info

test = dscompszipapt4append1['Outdoor Info']
test1 = test.str.split(',',expand=True)
test2 = list(test1.values.flatten())
test3 = [x for x in test2 if x != None]
test4 = [str(x) for x in test3]
test5 = [x.strip() for x in test4]
Counter(test5).most_common()

#('Gated', 383),
# ('nan', 356),
# ('Grill', 337),
# ('Sundeck', 326),
# ('Courtyard', 288),
# ('Picnic Area', 280),
# ('Balcony', 119),
# ('Rooftop Lounge', 91),
# ('Cabana', 81),
# ('Fenced Lot', 61),
# ('Patio', 49),
# ('Waterfront', 26),
# ('Pond', 18),
# ('Lake Access', 16),
# ('Zen Garden', 15),
# ('Barbecue/Grill', 14),
# ('Yard', 13),
# ('Barbecue Area', 11),
# ('Deck', 6),
# ('Porch', 4),
# ('Garden', 3),
# ('Lawn', 2),
# ('Boat Docks', 2)  

catvar = pandas.DataFrame(index=dscompszipapt4append1.index)

cat3('Outdoor Info',"Gated",'gated')
cat3('Outdoor Info',"Grill",'grill')
cat3('Outdoor Info',"Sundeck",'sundeck')
cat3('Outdoor Info',"Courtyard",'courtyard')
cat3('Outdoor Info',"Picnic_Area",'picnic area')
cat3('Outdoor Info',"Rooftop_Lounge",'rooftop lounge')
cat3('Outdoor Info',"Cabana",'cabana')

dscompszipapt4append1 = pandas.merge(dscompszipapt4append1, catvar, left_index=True, right_index=True)      

#pandas.crosstab(index=catvar["Business_Center"],  columns="count")
#(dscompszipapt4append1.isnull().sum()/len(dscompszipapt4append1.index)*100)

#Export the final file


pandas.DataFrame(dscompszipapt4append1.columns).to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/PreImpute_alldata.csv')
dscompszipapt4append2 = dscompszipapt4append1.drop(['useCodenew','Monthly Fees','One Time Fees','Pet Policy','Parking','Gym',
                                                    'Kitchen_x','Amenities','Features','Living Space','Lease Info','Services',
                                                    'Indoor Info','Outdoor Info','zestimate_amount_final','lastSoldPrice_final',
                                                    'source_final','Rent_parking','Cat_Rent','Dog_Rent','Fee_Application',
                                                    'Deposit_Cat','Deposit_Dog'],axis=1)

dscompszipapt4append2.to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Final Datasets/PreImpute_finaldata.csv')

