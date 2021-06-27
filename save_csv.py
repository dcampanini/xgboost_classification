#%%
import pandas as pd

data = {'Product': ['Desktop Computer','Tablet','Printer','Laptop'],
        'Price': [850,200,150,1300]
        }

df = pd.DataFrame(data, columns= ['Product', 'Price'])
df.to_csv(r'/home/diego/teco/hola.csv', 
        index = False, header=True)

print (df)

#%%
reclama1=pd.read_csv('/home/diego/teco/train_features.csv',
            low_memory=False)
reclama2=pd.read_csv('/home/diego/teco/test_features_junio.csv',
            low_memory=False)