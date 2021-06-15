#%%
import pandas as pd
import numpy as np
import time
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
# %%  Entrenar modelo 
# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
#%% fit model no training data
model = XGBClassifier()
model.fit(x_train, y_train)

#%% make predictions for test data
y_pred = model.predict(x_train)
#%%
predictions = [round(value) for value in y_pred]
#%% evaluate predictions
accuracy = accuracy_score(y_train, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('Fin ejecucion modelo')
# %%
elapsed = (end - start)
print('Tiempo en segundos',elapsed,'Tiempo en minutos',elapsed/60)
# %%
