import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from pprint import pprint

train=pd.read_csv(r'C:\Users\khair\Downloads\Train.csv')
test=pd.read_csv(r'C:\Users\khair\Downloads\Test.csv')
print(train)
print(test)

print(train.head())
print(train.tail())
print(test.head())
print(test.tail())
print(train.shape)
print(test.shape)
print(train.dtypes)
print(test.dtypes)
print(train.describe())
print(test.describe())

train['fare amount']=pd.to_numeric(train['fare_amount'],errors='coerce')
print(train['fare amount'])
print(train.dtypes)
print(train.shape)
print(train.dropna(subset=['pickup_datetime']))

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format='%d-%m-%Y %H:%M', errors='coerce')

train['year'] = train['pickup_datetime'].dt.year
train['month'] = train['pickup_datetime'].dt.month
train['Date'] = train['pickup_datetime'].dt.day
train['Day'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].dt.hour
train['Minute'] = train['pickup_datetime'].dt.minute

print(train['Minute'])
print(train['pickup_datetime'])
print(train.dtypes)
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], format='%d-%m-%Y %H:%M', errors='coerce')

test['year'] = test['pickup_datetime'].dt.year
test['month'] = test['pickup_datetime'].dt.month
test['Date'] = test['pickup_datetime'].dt.day
test['Day'] = test['pickup_datetime'].dt.dayofweek
test['Hour'] = test['pickup_datetime'].dt.hour
test['Minute'] = test['pickup_datetime'].dt.minute

print(test['Minute'])
print(test['pickup_datetime'])
print(test.dtypes)

train=train.drop(train[train['pickup_datetime'].isnull()].index,axis=0)
print(train.shape)
print(train['pickup_datetime'].isnull().sum())
print(train['passenger_count'].describe())

train=train.drop(train[train['passenger_count']>6].index,axis=0)
train=train.drop(train[train['passenger_count']==0].index,axis=0)
print(train)
print(train)
print(train['passenger_count'].describe())

print(train['passenger_count'].sort_values(ascending=True))

train=train.drop(train[train['passenger_count']==0.12].index,axis=0)
print(train.shape)

print(train['passenger_count'].sort_values(ascending=False))

print(Counter(train['fare_amount']<0))

train=train.drop(train[train['fare_amount']<0].index,axis=0)
print(train)
print(train.shape)
print(train['fare_amount'].min())

train=train.drop(train[train['fare_amount']<1].index,axis=0)
print(train.shape)
train=train.drop(train[train['fare_amount']>454].index,axis=0)
print(train.shape)

train=train.drop(train[train['fare_amount'].isnull()].index,axis=0)
print(train.shape)
print(train['fare_amount'].isnull().sum())
print(train['fare_amount'].describe())

print(train[train['pickup_latitude']<-90])
print(train[train['pickup_latitude']>90])

train=train.drop((train[train['pickup_latitude']<-90]).index,axis=0)
print(train)
train=train.drop((train[train['pickup_latitude']>90]).index,axis=0)
print(train)
print(train.shape)
print(train[train['pickup_longitude']<-180])

print(train[train['pickup_longitude']>180])
print(train[train['dropoff_latitude']<-90])

print(train[train['dropoff_latitude']>90])
print(train[train['dropoff_longitude']<-180])
print(train[train['dropoff_longitude']>180])

print(train.shape)
print(train.isnull().sum())
print(test.isnull().sum())

from math import radians,cos,sin,asin,sqrt
def haversine(row):
    lon1, lat1, lon2, lat2 = map(radians, row)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(h))
    km = 6371 * c
    return km

train['distance']=train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)
print(train['distance'])
test['distance']=test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)
print(test['distance'])
print(train.head())
print(test.head())
print(train.nunique())
print(test.nunique())

print(train['distance'].sort_values(ascending=False))
print(Counter(train['distance']==0))
print(Counter(test['distance']==0))
print(train[train['distance']<25])

train=train.drop(train[train['distance']==0].index,axis=0)
print(train)
print(train.shape)
test=test.drop(test[test['distance']==0].index,axis=0)
print(test)
print(test.shape)

drop_columns=['fare amount','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','Minute']
train=train.drop(drop_columns,axis=1)
print(train)
print(train.head())
print(train.dtypes)


drop_columns=['pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','Minute']
test=test.drop(drop_columns,axis=1)
print(test)
print(test.head())
print(test.dtypes)

plt.figure(figsize=(15,7))
sns.countplot(x='passenger_count',data=train)
plt.show()
plt.close()

plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'],y=train['fare_amount'],s=10)
plt.xlabel('No. of passenger')
plt.ylabel('fare')
plt.show()
plt.close()

plt.figure(figsize=(15,7))
plt.scatter(x=train['Date'],y=train['fare_amount'],s=10)
plt.xlabel('Date')
plt.ylabel('fare')
plt.show()
plt.close()

plt.figure(figsize=(15,7))
train.groupby(train['Hour'])['Hour'].count().plot(kind='bar')
plt.show()
plt.close()

plt.figure(figsize=(15,7))
plt.scatter(x=train['Hour'],y=train['fare_amount'],s=10)
plt.xlabel('Hour')
plt.ylabel('fare')
plt.show()
plt.close()

plt.figure(figsize=(15,7))
sns.countplot(x='Day',data=train)
plt.show()
plt.close()

plt.figure(figsize=(15,7))
plt.scatter(x=train['Day'],y=train['fare_amount'],s=10)
plt.xlabel('Hour')
plt.ylabel('fare')
plt.show()
plt.close()

plt.figure(figsize=(15,7))
plt.scatter(x=train['distance'],y=train['fare_amount'],c='g')
plt.xlabel('Distance')
plt.ylabel('fare')
plt.show()
plt.close()

for i in ['fare_amount', 'distance']:
    print(i)
    sns.histplot(train[i], bins='auto', color='green', kde=True)
    plt.title('Distribution for Variable ' + i)
    plt.ylabel('Density')
    plt.show()
    plt.close()
    
train['fare_amount']=np.log1p(train['fare_amount'])
train['distance']=np.log1p(train['distance'])
print(train['fare_amount'])
print(train['distance'])

for i in ['fare_amount', 'distance']:
    print(i)
    sns.histplot(train[i], bins='auto', color='green', kde=True)
    plt.title('Distribution for Variable ' + i)
    plt.ylabel('Density')
    plt.show()
    plt.close()

for i in ['distance']:
    print(i)
    sns.histplot(test[i], bins='auto', color='green', kde=True)
    plt.title('Distribution for Variable ' + i)
    plt.ylabel('Density')
    plt.show()
    plt.close()

test['distance']=np.log1p(test['distance'])
print(test['distance'])
for i in ['distance']:
    print(i)
    sns.histplot(test[i], bins='auto', color='green', kde=True)
    plt.title('Distribution for Variable ' + i)
    plt.ylabel('Density')
    plt.show()
    plt.close()
    
X_train,X_test,y_train,y_test=train_test_split(train.iloc[:,train.columns!='fare_amount'],train.iloc[:,0],test_size=0.20,random_state=1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

fit_LR=LinearRegression().fit(X_train,y_train)
print(fit_LR)
pred_train_LR=fit_LR.predict(X_train)
print(pred_train_LR)
pred_test_LR=fit_LR.predict(X_test)
print(pred_test_LR)
RMSE_test_LR=np.sqrt(mean_squared_error(y_test,pred_test_LR))
RMSE_train_LR=np.sqrt(mean_squared_error(y_train,pred_train_LR))
print('Root mean Squared Error For Training data= '+str(RMSE_train_LR))
print('Root mean Squared Error For Test data= '+str(RMSE_test_LR))

from sklearn.metrics import r2_score
print(r2_score(y_train,pred_train_LR))
print(r2_score(y_test,pred_test_LR))

fit_DT=DecisionTreeRegressor(max_depth=2).fit(X_train,y_train)
print(fit_DT)
pred_train_DT=fit_DT.predict(X_train)
print(pred_train_DT)
pred_test_DT=fit_DT.predict(X_test)
print(pred_test_DT)

RMSE_train_DT=np.sqrt(mean_squared_error(y_train,pred_train_DT))
RMSE_test_DT=np.sqrt(mean_squared_error(y_test,pred_test_DT))
print('Root mean Squared Error For Training data= '+str(RMSE_train_DT))
print('Root mean Squared Error For Test data= '+str(RMSE_test_DT))
from sklearn.metrics import r2_score
print(r2_score(y_train,pred_train_DT))
print(r2_score(y_test,pred_test_DT))


fit_RF = RandomForestRegressor(n_estimators=200, random_state=1).fit(X_train, y_train)
print(fit_RF)
pred_train_RF=fit_RF.predict(X_train)
print(pred_train_RF)
pred_test_RF=fit_RF.predict(X_test)
print(pred_test_RF)

RMSE_train_RF=np.sqrt(mean_squared_error(y_train,pred_train_RF))
RMSE_test_RF=np.sqrt(mean_squared_error(y_test,pred_test_RF))
print('Root mean Squared Error For Training data= '+str(RMSE_train_RF))
print('Root mean Squared Error For Test data= '+str(RMSE_test_RF))
from sklearn.metrics import r2_score
print(r2_score(y_train,pred_train_RF))
print(r2_score(y_test,pred_test_RF))

fit_GB=GradientBoostingRegressor().fit(X_train,y_train)
print(fit_GB)
pred_train_GB=fit_GB.predict(X_train)
print(pred_train_GB)
pred_test_GB=fit_GB.predict(X_test)
print(pred_test_GB)
RMSE_train_GB=np.sqrt(mean_squared_error(y_train,pred_train_GB))
RMSE_test_GB=np.sqrt(mean_squared_error(y_test,pred_test_GB))
print('Root mean Squared Error For Training data= '+str(RMSE_train_GB))
print('Root mean Squared Error For Test data= '+str(RMSE_test_GB))
from sklearn.metrics import r2_score
print(r2_score(y_train,pred_train_GB))
print(r2_score(y_test,pred_test_GB))

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(random_state=42)
print(rf)
from pprint import pprint
print('Parameter currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV

rand_grid = {
    'n_estimators': list(range(1, 10, 2)),
    'max_depth': list(range(1, 100, 2))
}

RRF = RandomForestRegressor(random_state=0)

randomcv_rf = RandomizedSearchCV(estimator=RRF,
                                 param_distributions=rand_grid,
                                 n_iter=5,
                                 cv=5,
                                 random_state=0)

randomcv_rf.fit(X_train, y_train)

best_model = randomcv_rf.best_estimator_
predictions_RRF = best_model.predict(X_test)

RRF_r2 = r2_score(y_test, predictions_RRF)
RRF_rmse = np.sqrt(mean_squared_error(y_test, predictions_RRF))

print('Random Search CV Random Forest Regressor Model Performance:')
print('Best parameters:', randomcv_rf.best_params_)
print('R_squared={:0.2f}.'.format(RRF_r2))
print('RMSE:', RRF_rmse)

from sklearn.model_selection import GridSearchCV

grid_params_rf = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20]
}

grid_rf = RandomForestRegressor(random_state=0)

gridcv_rf = GridSearchCV(estimator=grid_rf,
                         param_grid=grid_params_rf,
                         cv=5,              
                         verbose=1,         
                         n_jobs=-1)             

gridcv_rf.fit(X_train, y_train)

best_model_rf = gridcv_rf.best_estimator_
predictions_grid_rf = best_model_rf.predict(X_test)

grid_rf_r2 = r2_score(y_test, predictions_grid_rf)
grid_rf_rmse = np.sqrt(mean_squared_error(y_test, predictions_grid_rf))

print('Grid Search CV Random Forest Regressor Model Performance:')
print('Best parameters:', gridcv_rf.best_params_)
print('R_squared = {:.2f}'.format(grid_rf_r2))
print('RMSE = {:.4f}'.format(grid_rf_rmse))
print(test.head())

from sklearn.model_selection import GridSearchCV

grid_params_rf = {
    'n_estimators': list(range(50, 201, 50)),
    'max_depth': [5, 10, 15, 20]
}

grid_rf = RandomForestRegressor(random_state=0)

gridcv_rf = GridSearchCV(estimator=grid_rf,
                         param_grid=grid_params_rf,
                         cv=5,
                         verbose=1,
                         n_jobs=-1)

gridcv_rf.fit(X_train, y_train)

best_model_rf = gridcv_rf.best_estimator_
predictions_grid_rf = best_model_rf.predict(X_test)

grid_rf_r2 = r2_score(y_test, predictions_grid_rf)
grid_rf_rmse = np.sqrt(mean_squared_error(y_test, predictions_grid_rf))

print('Grid Search CV Random Forest Regressor Model Performance:')
print('Best parameters:', gridcv_rf.best_params_)
print('R_squared = {:0.2f}'.format(grid_rf_r2))
print('RMSE = {:.4f}'.format(grid_rf_rmse))


gb=GradientBoostingRegressor(random_state=42)
print(gb)
from pprint import pprint
print('Parameters currently in use:\n')
print(gb.get_params())

from sklearn.model_selection import RandomizedSearchCV

rand_grid = {
    'n_estimators': list(range(1, 10, 2)),
    'max_depth': list(range(1, 100, 2))
}

gb= GradientBoostingRegressor(random_state=0)

randomcv_gb = RandomizedSearchCV(estimator=gb,
                                 param_distributions=rand_grid,
                                 n_iter=5,
                                 cv=5,
                                 random_state=0)

randomcv_gb.fit(X_train, y_train)

best_model = randomcv_gb.best_estimator_
predictions_GB = best_model.predict(X_test)

GB_r2 = r2_score(y_test, predictions_GB)
GB_rmse = np.sqrt(mean_squared_error(y_test, predictions_RRF))

print('Random Search CV Gradient Boosting Regressor Model Performance:')
print('Best parameters:', randomcv_gb.best_params_)
print('R_squared={:0.2f}.'.format(GB_r2))
print('RMSE:', GB_rmse)


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

grid_params = {
    'n_estimators': list(range(1, 10, 2)),
    'max_depth': list(range(1, 100, 2))
}

GRF = RandomForestRegressor(random_state=0)

gridcv_rf = GridSearchCV(estimator=GRF,
                         param_grid=grid_params,
                         cv=5)

gridcv_rf.fit(X_train, y_train)

best_model = gridcv_rf.best_estimator_
predictions_GRF = best_model.predict(X_test)

GRF_r2 = r2_score(y_test, predictions_GRF)
GRF_rmse = np.sqrt(mean_squared_error(y_test, predictions_GRF))

print('Grid Search CV Random Forest Regressor Model Performance:')
print('Best parameters:', gridcv_rf.best_params_)
print('R_squared={:0.2f}.'.format(GRF_r2))
print('RMSE:', GRF_rmse)

from sklearn.ensemble import GradientBoostingRegressor

grid_params_gb = {
    'n_estimators': list(range(50, 201, 50)),
    'max_depth': [3, 4, 5, 6],                
    'learning_rate': [0.01, 0.05, 0.1]      
}

GRB = GradientBoostingRegressor(random_state=0)

gridcv_gb = GridSearchCV(estimator=GRB,
                         param_grid=grid_params_gb,
                         cv=5,
                         verbose=1,
                         n_jobs=-1)

gridcv_gb.fit(X_train, y_train)

best_model_gb = gridcv_gb.best_estimator_
predictions_GRB = best_model_gb.predict(X_test)

GRB_r2 = r2_score(y_test, predictions_GRB)
GRB_rmse = np.sqrt(mean_squared_error(y_test, predictions_GRB))

print('Grid Search CV Gradient Boosting Regressor Model Performance:')
print('Best parameters:', gridcv_gb.best_params_)
print('R_squared={:0.2f}.'.format(GRB_r2))
print('RMSE:', GRB_rmse)

final_predicted_fares = best_model_rf.predict(test)

final_predicted_fares = np.expm1(final_predicted_fares)

original_test = pd.read_csv(r'C:\Users\khair\Downloads\Test.csv')

original_test = original_test.loc[test.index]
original_test['predicted_fare_amount'] = final_predicted_fares

output_path = r'C:\Users\khair\Downloads\Final_Test_With_Predictions.csv'
original_test.to_csv(output_path, index=False)

print(" Predicted fares added and saved as:", output_path)
















































