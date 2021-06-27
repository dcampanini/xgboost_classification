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
def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result
#%%
def extract_features(num_pnm_data,fecha_macs,pnm_data):
    #%% generar arreglo para el set de entrenamiento
    field_num=3+3*num_pnm_data # =date+mac+clase + num_metrica*num_pnm_data
    set1= pd.DataFrame(0, index=range(len(fecha_macs)), columns=range(field_num))
    dictionary_list = []
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
            set1.iloc[i,-1]=0

        # % calcular mean y var
        mean=mac_data.iloc[:,3:16].mean()
        var=mac_data.iloc[:,3:16].var()
        # %% calcular weighted moving average (wma)
        weights=np.flip(np.arange(1,len(mac_data)+1))
        wma=mac_data.iloc[:,3:16].apply(lambda x: np.dot(x,weights)/sum(weights))
        #%% guardar mac,fecha,metricas en un dict
    
        #%% guardar fecha y mac_address
        pdb.set_trace()
        set1.iloc[i,0:2]=mac_data.iloc[0,[0,2]]
        #%% guardar mean, var, wma, mean-wma en set1
        set1.iloc[i,2:2+len(mean)]=mean
        set1.iloc[i,2+len(mean):2+2*len(mean)]=var
        set1.iloc[i,2+2*len(mean):2+3*len(mean)]=wma
        #set1.iloc[i,2+3*len(mean):2+4*len(mean)]=mean-wma
    # retornar arreglo con las features calculadas
    return set1

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
train1=train.sample(8000)
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
num_pnm_data=13
train1=extract_features(num_pnm_data,fecha_macs_train,train1)
#test1=extract_features(num_pnm_data,fecha_macs_test,test1)
print('Fin ejecucion feature engineering')
end = time.time()
print(end - start)
# %%
import random
start_time = time.time()
dictionary_list = []
for i in range(0, 10, 1):
    dictionary_data = {k: random.random() for k in range(30)}
    dictionary_list.append(dictionary_data)

df_final = pd.DataFrame.from_dict(dictionary_list)

end_time = time.time()
print('Execution time = %.6f seconds' % (end_time-start_time))