#%% 
import pickle
import pathlib
from datetime import datetime
#%% save model
today = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
# save model to file
pickle.dump(model, open(dir.joinpath(today+'.pickle.dat'), "wb"))
#%%
# load model from file
loaded_model = pickle.load(open(dir.joinpath(today+'.pickle.dat'), "rb"))
# make predictions for test data
y_pred = loaded_model.predict(x_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))