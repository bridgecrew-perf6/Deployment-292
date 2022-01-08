#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.api as sm
import streamlit as st


st.title ('Multi-Linear Model Deployment:- Predict the Mileage of car ' )

st.sidebar.header ('Cars Input Paramater ')

def cars_input_features():
    VOL = st.sidebar.number_input ('Insert Volume (pounds)')
    SP = st.sidebar.number_input ('Insert Speed (miles per hrs)')
    HP = st.sidebar.number_input ('Insert Horse-Power ')
    
    data = { 'VOL' : VOL,
             'SP' : SP,
             'HP' : HP,
             'loghp' :np.log(HP)
           }
    
    features = pd.DataFrame(data, index = [1] )
    return features

df = cars_input_features()
st.subheader('Cars Input Parameter ')
st.write(df)



# build our model 



cars = pd.read_csv("C:\\Users\\RUSHIKESH\\Downloads\\Cars.csv")



# model = 1

model = smf.ols('MPG~HP+SP+VOL+WT', data=cars).fit()


# model = 2 

model2 =smf.ols('MPG~VOL+SP+HP',data=cars).fit()




# remove 70th and 76th row

cars1 = cars.drop(cars.index[[70,76]],axis=0).reset_index() 
cars1 = cars1.drop(["index"],axis=1) 


# again remove 76th and 77th row

cars2 = cars1.drop(cars1.index[[76,77]], axis=0).reset_index()
cars2 = cars2.drop(['index'],axis=1)



# model = 3

model3 = smf.ols('MPG~HP+SP+VOL',data=cars2).fit()
model3.summary()



# model=4

cars2['loghp']=np.log(cars2.HP)

model4 = smf.ols('MPG~loghp+VOL+SP',data=cars2).fit()


# predict mileage for our given inputs

Predicted_mileage = model4.predict(df)


# display the output

st.subheader ('Predicted Mileage')
st.write (Predicted_mileage)














