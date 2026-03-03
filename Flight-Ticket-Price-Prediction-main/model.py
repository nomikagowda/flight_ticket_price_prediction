import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


df = pd.read_excel(r"C:\Users\yagat\Downloads\flight\Flight-Ticket-Price-Prediction-main\FlightFare_Dataset.xlsx")
print(df.shape)
df.head()



# Handling Date_of_Journey Column

df["journey_date"]=pd.to_datetime(df["Date_of_Journey"], format= "%d/%m/%Y").dt.day
df["journey_Month"]=pd.to_datetime(df["Date_of_Journey"], format= "%d/%m/%Y").dt.month
df.drop(["Date_of_Journey"],axis=1, inplace=True)
df.head()



# Handling Dep_Time Column

df["Dep_hour"]=pd.to_datetime(df["Dep_Time"]).dt.hour
df["Dep_minute"]=pd.to_datetime(df["Dep_Time"]).dt.minute
df.drop(["Dep_Time"], axis=1,inplace=True)
df.head()



#Handling Arrival_time column

df["Arrival_hour"]= pd.to_datetime(df.Arrival_Time).dt.hour
df["Arrival_minute"]= pd.to_datetime(df.Arrival_Time).dt.minute
df.drop(["Arrival_Time"], axis=1, inplace=True)

df.head()


df.Additional_Info.value_counts()



#Handling Duration column

dur= list(df.Duration)

for i in range(len(dur)):
    if len(dur[i].split()) != 2:
        if "h" in dur[i]:
            dur[i] = dur[i].strip() + " 0m"   
        else:
            dur[i] = "0h " + dur[i] 
            

Duration_hour=[]
Duration_minute=[]

for i in dur:
    
    a,b =i.split(sep="h")
    Duration_hour.append(int(a))
    m=b.strip()
    Duration_minute.append(int(m[0:len(m)-1]))

df["Duration_hour"]=Duration_hour
df["Duration_minute"]=Duration_minute
df.drop("Duration", axis=1, inplace=True)
df.head()


df["Airline"].value_counts()


air_count=df["Airline"].value_counts()
check=list(air_count.index[0:8])
type(check)


#Handling Airline Column

airline=df["Airline"]
new_airline=[]


for i in range(airline.shape[0]):
    if airline[i] in check:
        new_airline.append(airline[i])
    else:
        new_airline.append("Other")
df["Airline"]=new_airline
df["Airline"].value_counts()



airplane_dummy = pd.get_dummies(df.Airline)
print(airplane_dummy.shape)
airplane_dummy.head()



#Handling Source Column

df.Source.value_counts()

source = pd.get_dummies(df.Source)
source.head()


#Handling Destination Column

df.Destination.value_counts()


destination =pd.get_dummies(df.Destination)
destination.head()


# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


Total_Stops = pd.get_dummies(df.Total_Stops)
Total_Stops.head()



df.drop(["Airline","Source","Destination","Total_Stops"],  axis=1, inplace=True)
df.head()


df=pd.concat([airplane_dummy, source,destination,Total_Stops,df],axis=1)
df.head()


x=df.loc[:,df.columns!="Price"]
y=df["Price"]


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x, y)

df_c = df.copy()
df_c.iloc[937:938,:]


import joblib
joblib.dump(model, 'model.pkl')

first_half_columns= df_c.columns[:4]
joblib.dump(first_half_columns, 'first_half_columns.pkl')

sec_half_columns=x.columns[-8:]
joblib.dump(sec_half_columns, 'sec_half_columns.pkl')

all_cols=x.columns[:25]
joblib.dump(all_cols, 'all_cols.pkl')


