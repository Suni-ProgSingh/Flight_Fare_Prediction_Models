# Exact export of notebook cells

# --- Notebook cell 1 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --- Notebook cell 2 ---
df=pd.read_excel('flight_price.xlsx')
df.tail()

# --- Notebook cell 3 ---
#basic info about data
df.info()
df.describe()

# --- Notebook cell 4 ---
#Feature Engineering - handle date of journey
df['date']=df['Date_of_Journey'].str.split('/').str[0]
df['month']=df['Date_of_Journey'].str.split('/').str[1]
df['year']=df['Date_of_Journey'].str.split('/').str[2]
df['Date']=df['date'].astype(int)
df['Month']=df['month'].astype(int)
df['Year']=df['month'].astype(int)

df.drop(['date','month','year'], axis=1, inplace=True)

# --- Notebook cell 5 ---
df.info()

# --- Notebook cell 6 ---
#Handling Arrival Time - split by : into hour and minute
#First - we remove the date part from time wherever there
df['Arrival_Time']=df['Arrival_Time'].apply(lambda x:x.split(' ')[0])
df['Arrival_Hour']=df['Arrival_Time'].str.split(':').str[0]
df['Arrival_Minutes']=df['Arrival_Time'].str.split(':').str[1]
df.head(2)
df['Arrival_Hour']=df['Arrival_Hour'].astype(int)
df['Arrival_Minutes']=df['Arrival_Minutes'].astype(int)

# --- Notebook cell 7 ---
df['Dep_Hour']=df['Dep_Time'].str.split(':').str[0]
df['Dep_Min']=df['Dep_Time'].str.split(':').str[1]
df['Dep_Hour']=df['Dep_Hour'].astype(int)
df['Dep_Min']=df['Dep_Min'].astype(int)
df.drop('Dep_Time',axis=1,inplace=True)

# --- Notebook cell 8 ---
#Categorical features - Total stops
df['Total_Stops']=df['Total_Stops'].map({'non-stop':0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4, np.nan:1})


# --- Notebook cell 9 ---
df[df['Total_Stops'].isnull()]

# --- Notebook cell 10 ---
df.drop('Route', axis=1, inplace=True)

# --- Notebook cell 11 ---
df.head()

# --- Notebook cell 12 ---
#Get duration
df['Duration_Hour']=df['Duration'].str.split(' ').str[0].str.split('h').str[0]

# --- Notebook cell 13 ---
df['Duration_Min']=df['Duration'].str.split(' ').str[1].str.split('m').str[0]
df['Duration_Min']

# --- Notebook cell 14 ---
df['Duration_Min']=df['Duration_Min'].replace(np.nan,0)

# --- Notebook cell 15 ---
df['Duration_Hour']=df['Duration_Hour'].str.split('m').str[0]

# --- Notebook cell 16 ---
df['DurationInMin']=df['Duration_Hour'].astype(int)*60 + df['Duration_Min'].astype(int)

# --- Notebook cell 17 ---
df.drop('Duration', axis=1, inplace=True)

# --- Notebook cell 18 ---
df.head()

# --- Notebook cell 19 ---
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()

# --- Notebook cell 20 ---
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

encoder = OneHotEncoder()

ohe_cols=['Airline','Source', 'Destination', 'Additional_Info']
encoded=encoder.fit_transform(df[ohe_cols]).toarray()

encoded_df=pd.DataFrame(encoded, columns=encoder.get_feature_names_out(ohe_cols))

encoded_df.index=df.index
df=pd.concat([df.drop(ohe_cols, axis=1), encoded_df], axis=1)


# --- Notebook cell 21 ---


# --- Notebook cell 22 ---
df.drop(['Date_of_Journey', 'Arrival_Time', 'Duration_Hour', 'Duration_Min'], axis=1, inplace=True)

# --- Notebook cell 23 ---
df.head()

# --- Notebook cell 25 ---
#Simple Linear Regression Model
from sklearn.model_selection import train_test_split
X=df.drop('Price', axis=1)
Y=df['Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# --- Notebook cell 26 ---
#Model training
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

lr=LinearRegression()
lr.fit(X_train, Y_train)

# --- Notebook cell 27 ---
#Model prediction and evaluation

Y_pred = lr.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test, Y_pred))
r2=r2_score(Y_test, Y_pred)
print("Linear Regression RMSE: ", rmse)
print("Linear Regression R2 Score: ", r2)

# --- Notebook cell 28 ---
#Ridge, Lasso Regression
from sklearn.linear_model import Ridge, Lasso
ridge=Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, Y_train)
Y_pred_ridge = ridge.predict(X_test)

lasso=Lasso(alpha=0.001, random_state=42)
lasso.fit(X_train,Y_train)
Y_pred_lasso = lasso.predict(X_test)


# --- Notebook cell 29 ---
rmse_ridge=np.sqrt(mean_squared_error(Y_test, Y_pred_ridge))
r2_ridge=r2_score(Y_test, Y_pred_ridge)
print("Ridge Regression RMSE: ", rmse_ridge)
print("Ridge Regression R2 Score: ", r2_ridge)

rmse_lasso=np.sqrt(mean_squared_error(Y_test, Y_pred_lasso))
r2_lasso=r2_score(Y_test, Y_pred_lasso)
print("Lasso Regression RMSE: ", rmse_lasso)
print("Lasso Regression R2 Score: ", r2_lasso)

# --- Notebook cell 31 ---
#Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test)

rmse_dt=np.sqrt(mean_squared_error(Y_test, Y_pred_dt))
r2_dt=r2_score(Y_test, Y_pred_dt)
print("Decision Tree Regressor RMSE: ", rmse_dt)
print("Decision Tree Regressor R2 Score: ", r2_dt)

# --- Notebook cell 32 ---
#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=5, max_features='log2', random_state=42, n_jobs=-1)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)

rmse_rf=np.sqrt(mean_squared_error(Y_test, Y_pred_rf))
r2_rf=r2_score(Y_test, Y_pred_rf)
print("Random Forest Regressor RMSE: ", rmse_rf)
print("Random Forest Regressor R2 Score: ", r2_rf)

# --- Notebook cell 34 ---
#Gradient Boosted Tree
from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.1, max_features='sqrt', random_state=42, min_samples_leaf=5)
gbr.fit(X_train, Y_train)
Y_pred_gbr=gbr.predict(X_test)

rmse_gbr=np.sqrt(mean_squared_error(Y_test, Y_pred_gbr))
r2_gbr=r2_score(Y_test, Y_pred_gbr)
print("Gradient Boosting Regressor RMSE: ", rmse_gbr)
print("Gradient Boosting Regressor R2 Score: ", r2_gbr)


# --- Notebook cell 35 ---
# Training performance
Y_train_pred = gbr.predict(X_train)

rmse_train = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
r2_train = r2_score(Y_train, Y_train_pred)

print("Training RMSE:", rmse_train)
print("Training R²:", r2_train)

#Trainig & Test r2 scores are not too different so its not overfitting instead giving good results

# --- Notebook cell 36 ---
#XGBoost
from xgboost import XGBRegressor
xg=XGBRegressor(n_estimators=800,          # balanced number of trees
    learning_rate=0.05,        # conservative step
    max_depth=4,               # enough complexity
    subsample=0.8,             # randomness to prevent overfit
    colsample_bytree=0.8,      # feature sampling
    min_child_weight=3,        # avoid too small leaves
    reg_alpha=0.1,            # small L1
    reg_lambda=2.0,            # standard L2
    random_state=42,
    n_jobs=-1 )
xg.fit(X_train, Y_train)
Y_pred_xg=xg.predict(X_test)

rmse_xg=np.sqrt(mean_squared_error(Y_test, Y_pred_xg))
r2_xg=r2_score(Y_test, Y_pred_xg)
print("XG Boosting Regressor RMSE: ", rmse_xg)
print("XG Boosting Regressor R2 Score: ", r2_xg)


# --- Notebook cell 37 ---
# Training performance for XGBoost
Y_train_pred_xg = xg.predict(X_train)

rmse_train_xg = np.sqrt(mean_squared_error(Y_train, Y_train_pred_xg))
r2_train_xg = r2_score(Y_train, Y_train_pred_xg)

print("Training RMSE:", rmse_train_xg)
print("Training R²:", r2_train_xg)
