import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

scaler = pickle.load(open('scaler.pkl','rb'))

column_names = ['Patient_id','Appointment_id','Gender','Scheduled_day','Appointment_day', 'Age', 'Neighbourhood', 'Scholarship', 'Hypertension',
            'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']


important_features = ['Age_baby',
 'Age_senior',
 'Age_teen',
 'Age_young_adult',
 'Alcoholism',
 'Awaiting_day',
 'Diabetes',
 'Gender_M',
 'SMS_received',
 'Scholarship']



@app.route('/')
def home():
    return render_template('index.html')
    
 
@app.route('/predict',methods=['POST'])
def predict():
    
    features = []
    for inp in request.form.values():
        features.append(inp)
    features.pop()
    
    test = pd.DataFrame(columns=column_names,data=np.array(features).reshape(1,-1))
    

    
    test['Patient_id'] = test['Patient_id'].astype(np.int64)
    test['Appointment_id']=test['Appointment_id'].astype(np.int64)
    test['Gender']=test['Gender'].astype(object)
    test['Scheduled_day']=test['Scheduled_day'].astype(object)
    test['Appointment_day']=test['Appointment_day'].astype(object)
    test['Age']=test['Age'].astype(np.int64)
    test['Neighbourhood']=test['Neighbourhood'].astype(object)
    test['Scholarship']=test['Scholarship'].astype(np.int64)
    test['Hypertension']=test['Hypertension'].astype(np.int64)
    test['Diabetes']=test['Diabetes'].astype(np.int64)
    test['Alcoholism']=test['Alcoholism'].astype(np.int64)
    test['Handicap']=test['Handicap'].astype(np.int64)
    test['SMS_received']=test['SMS_received'].astype(object)

    test['Scheduled_day'] = pd.to_datetime(test['Scheduled_day']).dt.date.astype('datetime64[ns]')
    test['Appointment_day'] = pd.to_datetime(test['Appointment_day']).dt.date.astype('datetime64[ns]')

    test['Awaiting_day']=(test['Appointment_day']-test['Scheduled_day']).dt.days

    test['day_of_appointment']=test['Appointment_day'].dt.weekday_name

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


    test.Age = test.Age.apply(age_conv)

    test.drop(['Patient_id','Appointment_id','Scheduled_day','Appointment_day','Neighbourhood'],axis=1,inplace=True)
    if test.Gender[0] == 'M':
         test['Gender_M']=1
    else:
         test['Gender_M']=0
         
    if test.Age[0]=='baby':
         test['Age_baby']=1
         test['Age_senior']=0
         test['Age_teen']=0
         test['Age_young_adult']=0
    elif test.Age[0]=='senior':
         test['Age_baby']=0
         test['Age_senior']=1
         test['Age_teen']=0
         test['Age_young_adult']=0
    elif test.Age[0]=='teen':
         test['Age_baby']=0
         test['Age_senior']=0
         test['Age_teen']=1
         test['Age_young_adult']=0
    elif test.Age[0]=='young_adult':
         test['Age_baby']=0
         test['Age_senior']=0
         test['Age_teen']=0
         test['Age_young_adult']=1
    else:
         test['Age_baby']=0
         test['Age_senior']=0
         test['Age_teen']=0
         test['Age_young_adult']=0
         
    if test.day_of_appointment[0]=='Monday':
         test['day_of_appointment_Monday']=1
         test['day_of_appointment_Saturday']=0
         test['day_of_appointment_Thursday']=0
         test['day_of_appointment_Tuesday']=0
         test['day_of_appointment_Wednesday']=0
     
    elif test.day_of_appointment[0]=='Saturday':
         test['day_of_appointment_Monday']=0
         test['day_of_appointment_Saturday']=1
         test['day_of_appointment_Thursday']=0
         test['day_of_appointment_Tuesday']=0
         test['day_of_appointment_Wednesday']=0
     
    elif test.day_of_appointment[0]=='Thursday':
         test['day_of_appointment_Monday']=0
         test['day_of_appointment_Saturday']=0
         test['day_of_appointment_Thursday']=1
         test['day_of_appointment_Tuesday']=0
         test['day_of_appointment_Wednesday']=0
    
    elif test.day_of_appointment[0]=='Tuesday':
         test['day_of_appointment_Monday']=0
         test['day_of_appointment_Saturday']=0
         test['day_of_appointment_Thursday']=0
         test['day_of_appointment_Tuesday']=1
         test['day_of_appointment_Wednesday']=0
         
    elif test.day_of_appointment[0]=='Wednesday':
         test['day_of_appointment_Monday']=0
         test['day_of_appointment_Saturday']=0
         test['day_of_appointment_Thursday']=0
         test['day_of_appointment_Tuesday']=0
         test['day_of_appointment_Wednesday']=1    
    else:
         test['day_of_appointment_Monday']=0
         test['day_of_appointment_Saturday']=0
         test['day_of_appointment_Thursday']=0
         test['day_of_appointment_Tuesday']=0
         test['day_of_appointment_Wednesday']=0   
         
    test = test[important_features]    
    test = scaler.fit_transform(test)
    
    ypred_prob = model.predict(test)
    
    if ypred_prob >= 0.3:
        ypred = 'No Show' # 1
    else:
        ypred = 'Show' # 0
    
    output = ypred
    
    return render_template('index.html',prediction_text = "Patient will {}".format(output))




if __name__ == '__main__':
    app.run(debug=True)