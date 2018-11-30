import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import plot
from sklearn.model_selection import train_test_split

################3
####### DESCRIPTIVE ANALYSIS##################################################
##########


#This data was extracted from: https://www.kaggle.com/ntnu-testimon/paysim1

#It is a synthetic dataset of mobile money transactions. 
#Each step represents an hour of simulation.
##This dataset is scaled down 1/4 of the original dataset which is presented in
#"PaySim: A financial mobile money simulator for fraud detection".

fraud_file_path = '../input/PS_20174392719_1491204439457_log.csv'
fraud_data = pd.read_csv(fraud_file_path)
fraud_data=fraud_data.dropna()
fraud_data = fraud_data.drop_duplicates(fraud_data.columns, keep='last')
print(fraud_data.columns)
fraud_data.describe()
Index(['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
       'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud',
       'isFlaggedFraud'],
      dtype='object')


plt.title(r'Transactions Without Fraud')
print(fraud_data[fraud_data.isFraud==0].type.value_counts().head() / len(fraud_data))
(fraud_data[fraud_data.isFraud==0].type.value_counts().head() / len(fraud_data)).plot.bar(label='noFraud')

#From the total transactions, just 0.12% were Fraud. This 0.12% is divide in 0.0647% in Cash out and 0.0644% Transfer.

print(fraud_data[fraud_data.isFraud==1].type.value_counts().head() / len(fraud_data))
plt.title('Transactions With fraud')
(fraud_data[fraud_data.isFraud==1].type.value_counts().head() / len(fraud_data)).plot.bar(legend='isFraud')

#3Number of frauds (or no frauds) per unit of time.

plt.title(r'Step frecuency')
ax1=fraud_data[fraud_data.isFraud==1].step.value_counts().sort_index().plot.line(label='fraud')
ax1=fraud_data[fraud_data.isFraud==0].step.value_counts().sort_index().plot.line(ax=ax1)
ax1.legend(["Fraud", "No fraud"])

plt.title(r'Close up Fraud per unit of times')
ax1=fraud_data[fraud_data.isFraud==1].step.value_counts().sort_index().plot.line()
ax1.legend([ "Fraud"])

plt.title(r'Fraud Flagged or no per unit of time')
ax1=fraud_data[(fraud_data.isFraud==1)&(fraud_data.isFlaggedFraud==1)].step.value_counts().sort_index().plot.line(label='fraud')
ax1=fraud_data[(fraud_data.isFraud==1)&(fraud_data.isFlaggedFraud==0)].step.value_counts().sort_index().plot.line(label='fraud')
ax1.legend(["Fraud Flagged as Fraud","Fraud Not Flagged as Fraud"])

ax1=fraud_data[fraud_data.isFraud==1].plot.scatter(x='oldbalanceOrg', y='newbalanceOrig',c='blue',title='Relation old balance new balance',label='Fraud')
fraud_data[fraud_data.isFraud==0].plot.scatter(x='oldbalanceOrg', y='newbalanceOrig',c='orange',label='No Fraud',ax=ax1)
ax1.set_xlabel("Old balance Origin")
ax1.set_ylabel("New Balance Origin")

ax1=fraud_data[fraud_data.isFraud==0].plot.scatter(x='oldbalanceDest', y='newbalanceDest',c='orange',label='No Fraud')
fraud_data[fraud_data.isFraud==1].plot.scatter(x='oldbalanceDest', y='newbalanceDest',c='blue',title='Relation old balance new balance',label='Fraud',ax=ax1)
ax1.set_xlabel("Old balance Destination")
ax1.set_ylabel("New Balance Destination")


g=sns.boxplot(x='type',y='amount',data=fraud_data,palette='rainbow',hue='isFraud')
g.set_yscale('log')

###########3
#####33
#####Here i finished with the descriptive analysis

fraud_data.type.unique()
fraud_data=pd.get_dummies(data=fraud_data, columns=['type'])
features = ['amount','oldbalanceOrg', 'newbalanceOrig',
            'type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER',
        'oldbalanceDest', 'newbalanceDest', 'isFraud']
data2 = fraud_data[features]
data2.describe()

########I studied how the correlations changed when I worked with different percentages of data.
##########I choosed percentages of: 1,5,10,20,50, And I used the function sample to slect randomly the data.

data2['amount']=np.log1p(data2['amount'])
data2['oldbalanceOrg']=np.log1p(data2['oldbalanceOrg'])
data2['newbalanceOrig']=np.log1p(data2['newbalanceOrig'])
data2['oldbalanceDest']=np.log1p(data2['oldbalanceDest'])
data2['newbalanceDest']=np.log1p(data2['newbalanceDest'])
percent=[1,5,10,20,50,100]
df = pd.DataFrame({})
for value in percent:
    dataselect= data2.sample(frac=value/100)
    corr = dataselect.corr()
    df[str(value/100)]=corr['isFraud']
   # print(corr['isFraud'])#.sort_values(ascending=False))

############3
######3  SPLIT DATA
#########
finaldata=data2.sample(frac=.2)
colormap = plt.cm.magma
plt.figure(figsize=(11,11))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data2.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
y = finaldata['isFraud']
X = finaldata.drop(['isFraud'], axis=1).values 
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
