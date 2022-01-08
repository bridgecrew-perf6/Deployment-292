#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sn
import streamlit as st
import numpy as np



st.title ('Simple linear Deployment: To find the risk of Cardio-Vascular Diseases')

st.sidebar.header (" Patient Waist Circumference " )

def patient_input ():
    Waist = st.sidebar.number_input('Insert Measurement')
    
    data = { 'Waist' : Waist ,
            'Waist_sq' : Waist*Waist
           }
    feature = pd.DataFrame(data,index=[0])
    return feature

df = patient_input()
st.subheader ('Patient Waist Circumference')
st.write(df)


#Build model


#dataset is ready

data = pd.read_csv("C:\\Users\\RUSHIKESH\\Downloads\\WC_AT.csv")

# inputs and outputs 

data.Waist = data.Waist
data['Waist_sq'] = data.Waist * data.Waist
data['log_AT'] = np.log(data.AT)



#build final model

model = smf.ols ('log_AT~Waist+Waist_sq', data = data).fit()



# create a prediction for our input 

prediction = np.exp(model.predict(df))



# display the output

st.subheader(' Patient Abdominal Adipose Tissue (AT) Area ')
st.write ( prediction )

































