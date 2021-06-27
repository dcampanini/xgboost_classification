#%%
import pandas as pd
import numpy as np
from numpy import loadtxt
import time
import pdb
import pathlib
import pickle
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

#%%
def extract_features(fecha_macs,pnm_data):
    #%% generar arreglo para el set de entrenamiento
    dictionary_list = []
    # %% calcular metricas y armar dataset con features
    for i in range(len(fecha_macs)):
        #i=3
        # extraer para una mac y fecha especifica, todos los datos disponibles
        # por cada hora, idealmente serian 24 filas 
        mac_data=pnm_data.loc[(pnm_data['MAC_ADDRESS']==fecha_macs.iloc[i][1]) 
        & (pnm_data['DATE_FH']==fecha_macs.iloc[i][0])]
        #% etiquetar 
        if mac_data['ESTADO'].iloc[0]=='CERRADO':
            label=1
        else:
            label=0
        # % calcular mean y var
        mean=mac_data.iloc[:,3:16].mean()
        var=mac_data.iloc[:,3:16].var()
        # %% calcular weighted moving average (wma)
        weights=np.flip(np.arange(1,len(mac_data)+1))
        wma=mac_data.iloc[:,3:16].apply(lambda x: np.dot(x,weights)/sum(weights))
        #%% guardar mac,fecha,features en un dict
        # fila = lista con las mac,fecha,features
        fila=mac_data.iloc[0,[0,2]].tolist()+ mean.tolist()+\
             var.tolist()+wma.tolist()+[label]
        # trasformar fila en un dictionary
        keys=[i for i in range(len(fila))]
        data={k: v for k,v in zip(keys,fila)}
        # append the dictionary to  dictionary_list
        dictionary_list.append(data)
    # retornar arreglo con las features calculadas
    return dictionary_list

#%% =============================================================================  
start = time.time()
print('Cargando datos')
reclama1=pd.read_csv('train/mac_reclama_dia_abril.csv',low_memory=False)
reclama2=pd.read_csv('train/mac_reclama_dia_mayo.csv',low_memory=False)
reclama3=pd.read_csv('train/mac_reclama_dia_junio.csv',low_memory=False)
noreclama1=pd.read_csv('train/mac_no_reclama_label_abril.csv',low_memory=False)
noreclama2=pd.read_csv('train/mac_no_reclama_label_mayo.csv',low_memory=False)
noreclama3=pd.read_csv('train/mac_no_reclama_label_junio.csv',low_memory=False)
df_train=[reclama2,noreclama2]
df_test=[reclama1,reclama3,noreclama1,noreclama3]
train=pd.concat(df_train)
test=pd.concat(df_test)
#%%
#sub1=df2
train1=train.sample(1000)
test1=test.sample(500)
#%% drop some columns
train1=train1.drop(['FECHA_AFECTACION_00'], inplace=False, axis=1)
test1=test1.drop(['FECHA_AFECTACION_00'], inplace=False, axis=1)
#pnm_data=train1
#%% extraer dataframe  con mac_address unicas
#macs=sub1_ft.loc[:,['MAC_ADDRESS']].drop_duplicates().sort_values(by=['MAC_ADDRESS'])
#%% extraer dataframe  con (fecha,mac_address)
fecha_macs_train=train1.loc[:,['DATE_FH',
            'MAC_ADDRESS']].drop_duplicates().sort_values(by=['MAC_ADDRESS','DATE_FH'])
fecha_macs_test=test1.loc[:,['DATE_FH',
            'MAC_ADDRESS']].drop_duplicates().sort_values(by=['MAC_ADDRESS','DATE_FH'])
# %%
start = time.time()
print('Inicio ejecucion feature engineering')
d1=extract_features(fecha_macs_train,train1)
#test1=extract_features(num_pnm_data,fecha_macs_test,test1)
test1 = pd.DataFrame.from_dict(d1)
print('Fin ejecucion feature engineering')
end = time.time()
print(end - start)

#%%
set3=test1.dropna()
#%% Entrenar modelo 
print('Inicio entrenamiento modelo')
x_test=set3.iloc[:,2:41]
y_test=set3.iloc[:,41]
# %%
