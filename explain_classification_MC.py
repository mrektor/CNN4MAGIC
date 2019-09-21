import pickle
from keras.models import load_model
from keras_radam import RAdam
import shap

def pickle_read(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
#%%
model.load_weights('/data4T/CNN4MAGIC/results/MC_classification/computed_data/one-epoch-MV2.h5')
#%%
background = val_gn[0]
#%%
test=model.predict(background[0])
#%%
e = shap.DeepExplainer(model, background[0])
#%%
print(hadron_test_gn[0][0].shape)
#%%
hadron_example=hadron_test_gn[0][0][:1]
#%%
print(hadron_example.shape)
#%%
import numpy as np
shap_values = e.shap_values(background[0])

#%%
from keras_explain.saliency import Saliency
#%%
explainer = Saliency(model, layer=None)
#%%
exp = explainer.explain(hadron_example[0], 0)