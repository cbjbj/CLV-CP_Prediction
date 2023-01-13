import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import make_classification
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

showWarningOnDirectExecution = false

st.title("Welcome to the Customer Value and Collection Period Prediction Web App!")

st.subheader('Please fill in your client information down below:')



Business_Nature = st.selectbox('Business_Nature', ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'na'))
st.code("Business Nature Category:\nA - AGRICULTURE, FORESTRY AND FISHING\nB - MINING AND QUARRYING\nC - MANUFACTURING\nD - ELECTRICITY, GAS, STEAM AND AIR CONDITIONING SUPPLY\nE - WATER SUPPLY; SEWERAGE, WASTE MANAGEMENT AND REMEDIATION ACTIVITIES\nF - CONSTRUCTION\nG - WHOLESALE AND RETAIL TRADE, REPAIR OF MOTOR VEHICLES AND MOTORCYCLES\nH - TRANSPORTATION AND STORAGE\nI - ACCOMODATION AND FOOD SERVICE ACTIVITIES\nJ - INFORMATION AND COMMUNICATION\nK - FINANCIAL AND INSURANCE /TAKAFUL ACTIVITIES\nL - REAL ESTATE ACTIVITIES\nM - PROFESSIONAL, SCIENTIFIC AND TECHNICAL ACTIVITIES\nN - ADMINISTRATIVE AND SUPPORT SERVICE ACTIVITIES\nO - PUBLIC ADMINISTRATION AND DEFENCE, COMPULSORY SOCIAL ACTIVITIES\nP - EDUCATION\nQ - HUMAN HEALTH AND SOCIAL WORK ACTIVITIES\nR - ARTS, ENTERTAINMENT AND RECREATION\nS - OTHER SERVICE ACTIVITIES\nna - Not Available/No Information")


Company_Type = st.selectbox('Company Type', ('Berhad', 'Community', 'Lawyer Firm', 'Limited', 'Limited Liability Partnership', 'Partnership', 'Sendirian Berhad (Private Limited Company)', 'Sole Proprietorship'))
st.caption("Fill in na if not available")

State = st.selectbox('State (Perlis is not included as there were no data about it', ('Kedah', 'Penang', 'Kuantan', 'Terengganu', 'Pahang', 'Selangor', 'Perak', 'KualaLumpur', 'Putrajaya', 'NegeriSembilan', 'Melaka', 'Johor', 'Labuan', 'Sabah', 'Sarawak', 'na'))
st.caption("Fill in na if not available")

payment = st.multiselect('Select payment methods that the client uses' , ('Cash', 'Cheque','OnlineBanking'))
st.caption("You can choose more than 1 option")


data = {'Business_Nature': Business_Nature,
        'Company_Type': Company_Type,
        'State': State}
dfpay = {'Payment_Type' : payment}

df = pd.DataFrame(data, index = [0])
dfpay = pd.DataFrame(dfpay)


dfpay1 = pd.DataFrame({'Payment_Type': [', '.join(dfpay['Payment_Type'].str.strip('"').tolist())]})

df1 = dfpay1.join(df)

df = pd.read_csv('streamlit_df.csv')
df.drop(['Organization_Code', 'Payment_Bank_Code', 'Postcode'], axis = 1, inplace = True)
dfx = df.filter(['Customer_value', 'Collection_Period'], axis = 1)


X = df.drop(['Collection_Period', 'Customer_value'], axis = 1)
df = pd.concat([df1, X], axis=0)
df = df.reset_index()
del df['index']

df.Payment_Type = df.Payment_Type.str.replace(' ', '')
df.Business_Nature = df.Business_Nature.str.replace(' ', '')

def unique_col(col):
    return ','.join(set(col.split(',')))

    
df.Payment_Type = df.Payment_Type.fillna('na')
df.Business_Nature = df.Business_Nature.fillna('na')

#Perform Function on these columns
df['Payment_Type'] = df.Payment_Type.apply(unique_col)
df['Business_Nature'] = df.Business_Nature.apply(unique_col)


# Payment Type
#Create dummies
payment = pd.concat([df.drop('Payment_Type', 1), df['Payment_Type'].str.get_dummies(sep = ',')],1)
#Choose only relevant columns
payment = payment.iloc[:, 3:6]
#Add Prefix
payment.columns = 'pay_' + payment.columns

# Business Nature
#Create dummies
ind = pd.concat([df.drop('Business_Nature', 1), df['Business_Nature'].str.get_dummies(sep = ',')],1)
#Choose only relevant columns
ind = ind.iloc[:, 3:22]
#Add Prefix
ind.columns = 'ind_' + ind.columns

# Company Type
#Create dummies
comp_type = pd.concat([df.drop('Company_Type', 1), df['Company_Type'].str.get_dummies(sep = ',')],1)
#Choose only relevant columns
comp_type = comp_type.iloc[:, 3:]
#Add Prefix
comp_type.columns = 'comp_type_' + comp_type.columns

#State
#Create dummies
state = pd.concat([df.drop('State', 1), df['State'].str.get_dummies(sep = ',')],1)
#Choose only relevant columns
state = state.iloc[:, 3:18]
#Add Prefix
state.columns = 'state_' + state.columns

df = df.join(payment)
df = df.join(ind)
df = df.join(comp_type)
df = df.join(state)

df.drop(['Payment_Type', 'Business_Nature', 'Company_Type', 'State'], axis = 1, inplace = True)


from sklearn.preprocessing import normalize
df1_scaled = normalize(df)
df1_scaled = pd.DataFrame(df1_scaled, columns=df.columns)

df = df1_scaled[:1]

from sklearn.preprocessing import scale

#Load in model
model_clv = pickle.load(open('RandomForest_CLV.pkl', 'rb'))
model_cp = pickle.load(open('RandomForest_CP.pkl', 'rb'))

clv_pred = model_clv.predict(df)
cp_pred = model_cp.predict(df)

prediction_clv = model_clv.predict(df)
prediction_cp = model_cp.predict(df)


clv = np.array([0, 1, 2])
cp = np.array([0, 1, 2])

st.title('The model predicted that your client will have:')

if clv[prediction_clv] == 0:
    st.subheader('Low Quality in Customer Value')
elif clv[prediction_clv] == 1:
    st.subheader('Medium Quality in Customer Value')
elif clv[prediction_clv] == 2:
    st.subheader('High Quality in Customer Value')

if cp[prediction_cp] == 0:
    st.subheader('Low Quality in Collection Period')
elif clv[prediction_clv] == 1:
    st.subheader('Medium Quality in Collection Period')
elif clv[prediction_clv] == 2:
    st.subheader('High Quality in Collection Period')


