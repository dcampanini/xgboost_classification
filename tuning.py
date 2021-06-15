#%%
# import pandas for data wrangling
import pandas as pd
import numpy as np
# import machine learning libraries
import xgboost as xgb
from sklearn.metrics import accuracy_score
# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import os
import time
#%%
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) 
# will list all files under the input directory

#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))



#%%

# Mis datos
#%%
start = time.time()
df1=pd.read_csv('mac_reclama_dia.csv',low_memory=False)
df2=pd.read_csv('mac_no_reclama_label.csv',low_memory=False)
df3=df2.append(df1)
#%%
#sub1=df3
sub1=df3.sample(1000)
#%% drop some columns
sub1=sub1.drop(['FECHA_AFECTACION_00'], inplace=False, axis=1)
pnm_data=sub1
#%% extraer dataframe  con mac_address unicas
#macs=sub1_ft.loc[:,['MAC_ADDRESS']].drop_duplicates().sort_values(by=['MAC_ADDRESS'])
#%% extraer dataframe  con (fecha,mac_address)
fecha_macs=sub1.loc[:,['DATE_FH',
            'MAC_ADDRESS']].drop_duplicates().sort_values(by=['MAC_ADDRESS','DATE_FH'])

#%%
def extract_features(num_pnm_data,fecha_macs,pnm_data):
    #%% generar arreglo para el set de entrenamiento
    field_num=3+4*num_pnm_data # =date+mac+clase + num_metrica*num_pnm_data
    set1= pd.DataFrame(0, index=range(len(fecha_macs)), columns=range(field_num))
    # %% calcular metricas y armar dataset de entrenamiento
    for i in range(len(fecha_macs)):
        #i=3
        # extraer para una mac y fecha especifica, todos los datos disponibles
        # por cada hora, idealmente serian 24 filas 
        mac_data=pnm_data.loc[(pnm_data['MAC_ADDRESS']==fecha_macs.iloc[i][1]) 
        & (pnm_data['DATE_FH']==fecha_macs.iloc[i][0])]
        #% etiquetar 
        if mac_data['ESTADO'].iloc[0]=='CERRADO':
            set1.iloc[i,-1]=1
        else:
            set1.iloc[i,-1]=2

        # % calcular mean y var
        mean=mac_data.iloc[:,3:16].mean()
        var=mac_data.iloc[:,3:16].var()
        # %% calcular weighted moving average (wma)
        weights=np.flip(np.arange(1,len(mac_data)+1))
        wma=mac_data.iloc[:,3:16].apply(lambda x: np.dot(x,weights)/sum(weights))
        #%% guardar fecha y mac_address
        set1.iloc[i,0:2]=mac_data.iloc[0,[0,2]]
        #%% guardar mean, var, wma, mean-wma en set1
        set1.iloc[i,2:2+len(mean)]=mean
        set1.iloc[i,2+len(mean):2+2*len(mean)]=var
        set1.iloc[i,2+2*len(mean):2+3*len(mean)]=wma
        set1.iloc[i,2+3*len(mean):2+4*len(mean)]=mean-wma
    # retornar arreglo con las features calculadas
    return set1


# %%
num_pnm_data=13
set1=extract_features(num_pnm_data,fecha_macs,pnm_data)
print('Fin ejecucion feature engineering')
#%%
set2=set1.dropna()
end = time.time()
#%% load data
#dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
#X = dataset[:,0:8]
#Y = dataset[:,8]
#%% split data into train and test sets
#seed = 7
#test_size = 0.33
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
#%% 
x_train=set2.iloc[:,2:54]
y_train=set2.iloc[:,54]
# %% 
# Split data into separate training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.3, 
                                random_state = 0)
#%%
# Initialize domain space for range of values
space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }
#%%
# define objective function
def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }    
#%%
# algoritmo de ptimizacion
trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)
print("The best hyperparameters are : ","\n")
print(best_hyperparams)