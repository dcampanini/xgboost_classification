# [START setup]
import datetime
import pandas as pd
import numpy as np
import pdb
from google.cloud import storage

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# TODO: REPLACE 'YOUR_BUCKET_NAME' with your GCS Bucket name.
BUCKET_NAME = 'teco_ai'
# [END setup]


# ---------------------------------------
# 1. Add code to download the data from GCS (in this case, using the publicly hosted data).
# AI Platform will then be able to use the data when training your model.
# ---------------------------------------
# [START download-data]
# Public bucket holding the census data
bucket = storage.Client().bucket('teco_ai')

# Path to the data inside the public bucket
blob1 = bucket.blob('mac_reclama_dia_mayo.csv')
blob2 = bucket.blob('mac_no_reclama_label_mayo.csv')
# Download the data
blob1.download_to_filename('mac_reclama_dia_mayo.csv')
blob2.download_to_filename('mac_no_reclama_label_mayo.csv')

# [END download-data]


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
        set1.iloc[i,0:2]=mac_data.iloc[0,[0,2]].values
        #%% guardar mean, var, wma, mean-wma en set1
        set1.iloc[i,2:2+len(mean)]=mean.values
        set1.iloc[i,2+len(mean):2+2*len(mean)]=var.values
        set1.iloc[i,2+2*len(mean):2+3*len(mean)]=wma.values
        #pdb.set_trace()
        set1.iloc[i,2+3*len(mean):2+4*len(mean)]=(mean-wma).values
    # retornar arreglo con las features calculadas
    return set1



# ---------------------------------------
# This is where your model code would go. Below is an example model using the census dataset.
# ---------------------------------------
# [START define-and-load-data]
# Define the format of your input data including unused columns (These are the columns from the census data files)


# Load the training census dataset
'''with open('./mac_reclama_dia_mayo.csv', 'r') as train_data2:
    reclama2= pd.read_csv(train_data2,low_memory=False)
with open('./mac_no_reclama_label_mayo.csv', 'r') as train_data1:
    noreclama2= pd.read_csv(train_data1,low_memory=False)'''
reclama2=pd.read_csv('mac_reclama_dia_mayo.csv',low_memory=False)
noreclama2=pd.read_csv('mac_no_reclama_label_mayo.csv',low_memory=False)
#noreclama2=reclama2
#pdb.set_trace()
#reclama2.head(5)
#noreclama2.head(5)

df_train=[reclama2,noreclama2]
train=pd.concat(df_train)
train=train.sample(1000)
train1=train.drop(['FECHA_AFECTACION_00'], inplace=False, axis=1)
print('train.head',train.head())
#%% extraer dataframe  con (fecha,mac_address)
fecha_macs_train=train1.loc[:,['DATE_FH',
            'MAC_ADDRESS']].drop_duplicates().sort_values(by=['MAC_ADDRESS','DATE_FH'])
# %%
print('Inicio ejecucion feature engineering')
num_pnm_data=13
train1=extract_features(num_pnm_data,fecha_macs_train,train1)
#test1=extract_features(num_pnm_data,fecha_macs_test,test1)
print('Fin ejecucion feature engineering')
#%%
set2=train1.dropna()
#set3=test1.dropna()
# Entrenar modelo 
#pdb.set_trace()
print('Inicio entrenamiento modelo')
x_train=set2.iloc[:,2:54]
y_train=set2.iloc[:,54]
print('train1',train1.shape)
print('set2',set2.shape)
print('x_train.shape= ',x_train.shape)
print('y_train.shape',y_train.shape)
#x_test=set3.iloc[:,2:54]
#y_test=set3.iloc[:,54]

#%% data otra
#df1 = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
#df2 =pd.DataFrame(np.ones(50))
#df3=pd.DataFrame(np.zeros(50))
#df_label=pd.concat([df2,df3])

#%% fit model on training data
model1 = XGBClassifier()
#model1 = XGBClassifier(scale_pos_weight=100)
model1.fit(x_train, y_train,verbose=True)
print('Fin entrenamiento modelo')


# Create the overall model as a single pipeline
pipeline = Pipeline([
    ('classifier', model1)
])
# [END create-pipeline]

# ---------------------------------------
# 2. Export and save the model to GCS
# ---------------------------------------
# [START export-to-gcs]
# Export the model to a file
model = 'model.joblib'
joblib.dump(pipeline, model)

# Upload the model to GCS
bucket = storage.Client().bucket(BUCKET_NAME)
blob = bucket.blob('{}/{}'.format(
    datetime.datetime.now().strftime('teco_%Y%m%d_%H:%M:%S'),
    model))
blob.upload_from_filename(model)
# [END export-to-gcs]