import pickle
from keras.models import load_model
# from keras_radam import RAdam
import shap

def pickle_read(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
#%%
prova = load_model('/data4T/CNN4MAGIC/results/MC_classification/computed_data/one-epoch-MV2.h5')
#%%

#%%
background = val_gn[0]
#%%
test=prova.predict(background[0])
#%%
e = shap.DeepExplainer(prova, background[0])
#%%
print(hadron_test_gn[0][0].shape)
#%%
hadron_example=val_gn[0][0][:1]
#%%
print(hadron_example.shape)
#%%
import numpy as np
shap_values = e.shap_values(background[0])
#%%
prova.summary()
#%%
from keras_explain.saliency import Saliency
#%%
explainer = Saliency(prova, layer=None)
#%%
exp = explainer.explain(hadron_example[0], 0)

#%%
last = prova.layers[-1].get_weights()
#%%
last[0].shape
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(13,8))
plt.hist(last[0], 80)
plt.xlabel('Last Dense')
plt.title(f'Distribution of the Weights of the last dense layer of the classifier. bias = {last[1][0]:.5f}')
plt.tight_layout()
plt.savefig('/data4T/CNN4MAGIC/results/MC_classification/plots/last_dense.png')
plt.close()