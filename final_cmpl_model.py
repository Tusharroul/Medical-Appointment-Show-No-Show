import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score,recall_score,accuracy_score,confusion_matrix,f1_score
from sklearn.metrics import precision_recall_curve,auc,roc_auc_score,roc_curve

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle

os.chdir("C:\\Users\\hp\\Documents\\R_and_PY_programming\\tushar\\binf_capstone_projects\\project2_healthcare\\Healthcare_deployment")

df = pd.read_csv("data.csv")

df.columns=['Patient_id','Appointment_id','Gender','Scheduled_day','Appointment_day', 'Age', 'Neighbourhood', 'Scholarship', 'Hypertension',
            'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received', 'No_show']

df_temp = df.copy()

df.info()

df['Patient_id'] = df['Patient_id'].astype('int64')
df['Scheduled_day'] = pd.to_datetime(df['Scheduled_day']).dt.date.astype('datetime64[ns]')
df['Appointment_day'] = pd.to_datetime(df['Appointment_day']).dt.date.astype('datetime64[ns]')

df['Awaiting_day']=(df['Appointment_day']-df['Scheduled_day']).dt.days
df=df[df.Awaiting_day >= 0]
df.Awaiting_day = np.log1p(df.Awaiting_day)

df['day_of_appointment']=df['Appointment_day'].dt.weekday_name

df = df[(df.Age >= 0) & (df.Age <= 110)]

def age_conv(x):
    if (x>=0) and (x<=10):
        return 'baby'
    elif  (x>=11) and (x<=20):
        return 'teen'
    elif  (x>=21) and (x<=40):
        return 'young_adult'
    elif  (x>=41) and (x<= 60):
        return 'adult'
    elif  (x>=61):
        return 'senior'


df.Age = df.Age.apply(age_conv)

df.drop(['Patient_id','Appointment_id','Scheduled_day','Appointment_day','Neighbourhood'],axis=1,inplace=True)
df = pd.get_dummies(df,drop_first=True)


X = df.drop(['No_show_Yes'], axis=1).copy()
y = df['No_show_Yes'].copy()



logreg = LogisticRegression()
rfe= RFE(logreg,15) # taking top 20 featrues

scaler = StandardScaler()
X  = scaler.fit_transform(X)
fit = rfe.fit(X,y)


X_pre = df.drop(['No_show_Yes'], axis=1).copy()
names=X_pre.columns


sort =sorted( zip(map(lambda x : round(x,4),rfe.ranking_),names) )

important_features = []
for i in range(10):
    important_features.append(sort[i][1])
    

X = df.drop(['No_show_Yes'], axis=1).copy()
y = df['No_show_Yes'].copy()
X = X[important_features]


scaler = StandardScaler()
X = scaler.fit_transform(X)

pickle.dump(scaler,open('scaler.pkl','wb'))

model = LogisticRegression()
model.fit(X,y)

pickle.dump(model,open('model.pkl','wb'))
# predict probability at 0.3

