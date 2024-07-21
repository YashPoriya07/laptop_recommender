#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('laptop_data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.duplicated().sum()


# In[7]:


df.isnull().sum()


# In[8]:


df=df.drop(columns='Unnamed: 0',axis=1)


# In[9]:


df.head()


# In[10]:


df['Ram']=df['Ram'].str.replace('GB','')
df['Weight']=df['Weight'].str.replace('kg','')


# In[11]:


df.head()


# In[12]:


df['Ram']=df['Ram'].astype('int32')
df['Weight']=df['Weight'].astype('float32')


# In[13]:


df.info()


# In[14]:


import seaborn as sns


# In[15]:


sns.distplot(df['Price'])


# In[16]:


df['Company'].value_counts().plot(kind='bar')


# In[17]:


sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[18]:


df['TypeName'].value_counts().plot(kind='bar')


# In[19]:


sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[20]:


sns.distplot(df['Inches'])


# In[21]:


sns.scatterplot(x=df['Inches'],y=df['Price'])


# In[22]:


df['ScreenResolution'].value_counts()


# In[23]:


df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[24]:


df.sample(5)


# In[25]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[26]:


sns.barplot(x=df['Touchscreen'],y=df['Price'])


# In[27]:


df['Ips']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[28]:


df.head()


# In[29]:


df['Ips'].value_counts().plot(kind='bar')


# In[30]:


sns.barplot(x=df['Ips'],y=df['Price'])


# In[31]:


new=df['ScreenResolution'].str.split('x',n=1,expand=True)


# In[32]:


df['X_res']=new[0]
df['Y_res']=new[1]


# In[33]:


df.head()


# In[34]:


df['X_res']=df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])  #regular expression


# In[35]:


df.head()


# In[36]:


df.info()


# In[37]:


df['X_res']=df['X_res'].astype('int')
df['Y_res']=df['Y_res'].astype('int')


# In[38]:


df.info()


# In[39]:


df['ppi']=(((df['X_res']**2+df['Y_res']**2))**0.5/df['Inches']).astype('float') 
#formula of ppi


# In[40]:


df=df.drop(columns='ScreenResolution',axis=1)


# In[41]:


df.head()


# In[42]:


df=df.drop(columns=['Inches','X_res','Y_res'],axis=1)


# In[43]:


df.head()


# In[44]:


df['Cpu'].value_counts()


# In[45]:


df['Cpu Name']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[46]:


df.head()


# In[47]:


def fetch_processor(text):
    if text=='Intel Core i7' or text=='Intel Core i5' or text=='Intel Core i3':
        return text
    else:
        if text.split()[0]=='Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[48]:


df['Cpu brand']=df['Cpu Name'].apply(fetch_processor)


# In[49]:


df.head()


# In[50]:


df['Cpu brand'].value_counts()


# In[51]:


sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[52]:


df=df.drop(columns=['Cpu','Cpu Name'],axis=1)


# In[53]:


df.head()


# In[54]:


df['Ram'].value_counts().plot(kind='bar')


# In[55]:


sns.barplot(x=df['Ram'],y=df['Price'])


# In[56]:


df['Memory'].value_counts()


# In[57]:


import pandas as pd

# Sample DataFrame

# Clean the "Memory" column
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')

# Split "Memory" column into "first" and "second"
new = df["Memory"].str.split("+", n=1, expand=True)
df["first"] = new[0]
df["first"] = df["first"].str.strip()

# Extract numeric values and storage type from "first" column
df[['first_value', 'storage_type']] = df['first'].str.extract(r'(\d+)([A-Za-z\s]+)')

# Convert "first_value" to integers
df['first_value'] = pd.to_numeric(df['first_value'], errors='coerce').fillna(0).astype(int)

# Identify storage types using lambda functions
df["Layer1HDD"] = df["storage_type"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["storage_type"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["storage_type"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["storage_type"].apply(lambda x: 1 if "Flash Storage" in x else 0)

# Extract numeric values and calculate based on storage types
df["first_value"] = df["first_value"].astype(int)
df["HDD"] = df["first_value"] * df["Layer1HDD"]
df["SSD"] = df["first_value"] * df["Layer1SSD"]
df["Hybrid"] = df["first_value"] * df["Layer1Hybrid"]
df["Flash_Storage"] = df["first_value"] * df["Layer1Flash_Storage"]

# Drop intermediate columns
df.drop(columns=['Memory', 'first', 'storage_type', 'first_value', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage'], inplace=True)



# In[58]:


df.head(5)


# In[59]:


df['HDD'].value_counts()


# In[60]:


df1=df.drop(columns=['Company','TypeName','Gpu','OpSys','Cpu brand'],axis=1)


# In[61]:


df1.corr()['Price']


# In[62]:


df=df.drop(columns=['Hybrid','Flash_Storage'],axis=1)


# In[63]:


df.head()


# In[64]:


df['Gpu'].value_counts()


# In[65]:


df['Gpu_brand']=df['Gpu'].apply(lambda x:x.split()[0])


# In[66]:


df.head()


# In[67]:


df['Gpu_brand'].value_counts()


# In[68]:


df=df[df['Gpu_brand']!='ARM']


# In[69]:


df['Gpu_brand'].value_counts()


# In[70]:


sns.barplot(x=df['Gpu_brand'],y=df['Price'],estimator=np.median)


# In[71]:


df=df.drop(columns='Gpu',axis=1)


# In[72]:


df.head()


# In[73]:


df['OpSys'].value_counts()


# In[74]:


sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')


# In[75]:


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[76]:


df['os']=df['OpSys'].apply(cat_os)


# In[77]:


df.head()


# In[78]:


df=df.drop(columns='OpSys',axis=1)


# In[79]:


sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='vertical')


# In[80]:


sns.distplot(df['Weight'])


# In[81]:


sns.scatterplot(x=df['Weight'],y=df['Price'])


# In[82]:


df1.corr()['Price']


# In[83]:


sns.heatmap(df1.corr())


# In[84]:


sns.distplot(df['Price'])


# In[85]:


sns.distplot(np.log(df['Price']))


# In[86]:


x=df.drop(columns='Price',axis=1)
y=np.log(df['Price'])


# In[87]:


x


# In[88]:


y


# In[89]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[90]:


x_train


# In[91]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.preprocessing import OneHotEncoder


# In[92]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# # Linear Regression

# In[93]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('M.A.E',mean_absolute_error(y_test,y_pred))



# # Lasso Regression

# In[94]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('M.A.E',mean_absolute_error(y_test,y_pred))



# # KNN

# In[95]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('M.A.E',mean_absolute_error(y_test,y_pred))


# # Decision Tree`

# In[96]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('M.A.E',mean_absolute_error(y_test,y_pred))


# # SVM

# In[97]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('M.A.E',mean_absolute_error(y_test,y_pred))


# # ExtraTrees

# In[98]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = ExtraTreesRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=None,
                              max_features=0.75,
                              max_depth=15)
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('M.A.E',mean_absolute_error(y_test,y_pred))


# # AdaBoost

# In[99]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=15,learning_rate=1.0)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('M.A.E',mean_absolute_error(y_test,y_pred))


# # Gradient Boost

# In[100]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('M.A.E',mean_absolute_error(y_test,y_pred))


# # XgBoost

# In[101]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('M.A.E',mean_absolute_error(y_test,y_pred))


# # Voting Regressor

# In[102]:


from sklearn.ensemble import VotingRegressor,StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')


rf = RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)
gbdt = GradientBoostingRegressor(n_estimators=100,max_features=0.5)
xgb = XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5)
et = ExtraTreesRegressor(n_estimators=100,random_state=3,max_samples=None,max_features=0.75,max_depth=10)

step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb',xgb), ('et',et)],weights=[5,1,1,1])

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Random Forest

# In[104]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 =RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('M.A.E',mean_absolute_error(y_test,y_pred))


# In[105]:


df.head()


# # Exporting the Model

# In[107]:


import pickle

pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[ ]:




