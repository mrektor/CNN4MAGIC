import pickle

from keras.models import load_model

from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

# %
# Load the data
BATCH_SIZE = 128
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                  want_golden=False,
                                                                  want_energy=True)
# %
model = load_model('/home/emariott/deepmagic/output_data/checkpoints/MobileNetV2-separation-big.hdf5')

y_pred_test = model.predict_generator(test_gn, workers=3, verbose=1)

dump_name = '/home/emariott/deepmagic/output_data/reconstructions/gamma_separation_test_predict_MobileNetV2-separation-big.pkl'
with open(dump_name, 'wb') as f:
    pickle.dump(y_pred_test, f)

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.hist(y_pred_test, bins=100, log=True)
plt.xlabel('Gammaness')
plt.ylabel('Counts')
plt.title(f'Gammaness of {y_pred_test.shape[0]} Point-Like MC')
plt.savefig('output_data/pictures/gammaness_pointlike.png')
plt.close()
