#%%
import pandas as pd
import numpy as np
from numpy import loadtxt
import time
import pathlib
import pickle
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt 
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
train1=train.sample(500)
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
print('Inicio ejecucion feature engineering')
num_pnm_data=13
train1=extract_features(num_pnm_data,fecha_macs_train,train1)
test1=extract_features(num_pnm_data,fecha_macs_test,test1)
print('Fin ejecucion feature engineering')
#%%
set2=train1.dropna()
set3=test1.dropna()
#%% =============================================================================  
# Entrenar modelo 
print('Inicio entrenamiento modelo')
x_train=set2.iloc[:,2:54]
y_train=set2.iloc[:,54]
x_test=set3.iloc[:,2:54]
y_test=set3.iloc[:,54]
#%% fit model on training data
model = XGBClassifier()
model.fit(x_train, y_train,verbose=True)
print('Fin entrenamiento modelo')
#%% make predictions for test data
y_pred = model.predict(x_test)
#%%
predictions = [round(value) for value in y_pred]
#%% evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('Fin ejecucion modelo')
# %%
end = time.time()
elapsed = (end - start)
print('Tiempo en segundos',elapsed,'Tiempo en minutos',elapsed/60)
#%% =============================================================================  
#%% save results and model
#make a directory
today = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
dir=pathlib.Path.cwd().joinpath('results'+today)
pathlib.Path(dir).mkdir(parents=True, exist_ok=True) 
#%% save real labels and predicted
y_test_np=y_test.to_numpy()
output=pd.DataFrame({'real':y_test,'pred':y_pred})
output.to_csv(dir.joinpath('results.csv'))
# save accuracy
np.savetxt(dir.joinpath('accuracy.txt'),np.resize(np.array(accuracy),(1,1)))
#%% save model
model.save_model(dir.joinpath(today+'.model'))
pickle.dump(model, open(dir.joinpath(today+'.pickle.dat'), "wb"))
# %% confusion matrix 
plot_confusion_matrix(model, x_test, y_test,
                    display_labels=['reclama', 'no reclama'],
                    cmap=plt.cm.Blues) 
plt.savefig(dir.joinpath('confusion_matrix.jpg'),dpi=300)
#plt.show()