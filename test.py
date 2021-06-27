from sklearn.externals import joblib
from google.cloud import storage
import datetime
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
#datos test
BUCKET_NAME = 'teco_ai'
# [START download-data]
# Public bucket holding the census data
bucket = storage.Client().bucket('teco_ai')

# Path to the data inside  bucket
blob1 = bucket.blob('mac_reclama_dia_junio.csv')
blob2 = bucket.blob('mac_no_reclama_label_junio.csv')
# Download the data
blob1.download_to_filename('mac_reclama_dia_junio.csv')
blob2.download_to_filename('mac_no_reclama_label_junio.csv')

# download model 
#teco_ai/teco_20210618_170420
blob3=bucket.blob('teco_20210618_170420/model.joblib')
blob3.download_to_filename('model.joblib')

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
            set1.iloc[i,-1]=0

        # % calcular mean y var
        mean=mac_data.iloc[:,3:16].mean()
        var=mac_data.iloc[:,3:16].var()
        # %% calcular weighted moving average (wma)
        weights=np.flip(np.arange(1,len(mac_data)+1))
        wma=mac_data.iloc[:,3:16].apply(lambda x: np.dot(x,weights)/sum(weights))
        #%% guardar fecha y mac_address
        set1.iloc[i,0:2]=mac_data.iloc[0,[0,2]].values
        #%% guardar mean, var, wma, mean-wma en set1
        set1.iloc[i,2:2+len(mean)]=mean.values
        set1.iloc[i,2+len(mean):2+2*len(mean)]=var.values
        set1.iloc[i,2+2*len(mean):2+3*len(mean)]=wma.values
        #pdb.set_trace()
        set1.iloc[i,2+3*len(mean):2+4*len(mean)]=(mean-wma).values
    # retornar arreglo con las features calculadas
    return set1

# cargar datos
reclama2=pd.read_csv('mac_reclama_dia_junio.csv',low_memory=False)
noreclama2=pd.read_csv('mac_no_reclama_label_junio.csv',low_memory=False)

df_test=[reclama2,noreclama2]
test=pd.concat(df_test)
test1=test.sample(1000)
test1=test1.drop(['FECHA_AFECTACION_00'], inplace=False, axis=1)

#%% extraer dataframe  con (fecha,mac_address)
fecha_macs_test=test1.loc[:,['DATE_FH',
            'MAC_ADDRESS']].drop_duplicates().sort_values(by=['MAC_ADDRESS',
            'DATE_FH'])

# %%
print('Inicio ejecucion feature engineering')
num_pnm_data=13
test1=extract_features(num_pnm_data,fecha_macs_test,test1)
print('Fin ejecucion feature engineering')
#%%
set3=test1.dropna()

# Entrenar modelo 
print('Inicio entrenamiento modelo')
x_test=set3.iloc[:,2:54]
y_test=set3.iloc[:,54]


# load and test model
loaded_model = joblib.load('model.joblib')
#result = loaded_model.score(x_test, y_test)
#print(result)

#%% make predictions for test data
y_pred = loaded_model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#%% precision and recall
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test, y_pred, average='macro')  
print("Precision:",precision)
print("Recall:",recall)

#%% confusion matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)
print(type(cm))

# save results  to a bucket
df=pd.DataFrame({'real':y_test,'pred':y_pred})
client = storage.Client()
bucket = client.get_bucket('teco_ai')
bucket.blob('upload_test/test.csv').upload_from_string(df.to_csv(), 'text/csv')