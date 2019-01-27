import numpy as np
from keras.models import load_model

from CNN4MAGIC.CNN_Models.BigData.utils import plot_angular_resolution
from CNN4MAGIC.Generator.gen_util import load_point_generator

# Loading the position data
BATCH_SIZE = 512
point_gn, position_te = load_point_generator(batch_size=BATCH_SIZE, want_position=True)

# Lima i dati perché prendendoli a batch te ne perdi qualcuno
position_te = np.array(position_te)
position_te_limato = position_te[:len(point_gn) * BATCH_SIZE]

# % Load the Model
net_name = 'MobileNetV2_4dense_position-big-2'
filepath = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/MobileNetV2_4dense_position-big-2.hdf5'
model = load_model(filepath)

# Make the predictions
print('Making predictions on test set...')
position_prediction = model.predict_generator(generator=point_gn, verbose=1, use_multiprocessing=True, workers=8)

# % Load the Energy so that you can make energy-binnings
point_gn, energy_te = load_point_generator(batch_size=BATCH_SIZE, want_energy=True)

energy_te = np.array(energy_te)
energy_te_limato = energy_te[:len(point_gn) * BATCH_SIZE]

# % Plot this resolution
# %%
plot_angular_resolution(position_te_limato, position_prediction, energy_te_limato, net_name=net_name + ' POINT-LIKE',
                        fig_folder='/home/emariott/deepmagic/CNN4MAGIC/Generator/position_pic')
# %%
import pickle

with open('/home/emariott/deepmagic/CNN4MAGIC/Generator/position_predictions/pos_pred' + net_name + '.pkl', 'wb') as f:
    pickle.dump(position_prediction, f)
# %%
from CNN4MAGIC.CNN_Models.BigData.utils import bin_data_mask, compute_theta
import matplotlib.pyplot as plt


def plot_angular_resolution(position_true, position_prediction, energy_true,
                            fig_folder='/data/code/CNN4MAGIC/Generator/position_pic', net_name=''):
    binned_values, bins, bins_masks = bin_data_mask(energy_true, 11)
    resolutions = []
    bin_medians = []

    for i, mask in enumerate(bins_masks):
        bin_pos = position_true[mask]
        bin_pred_pos = position_prediction[mask]
        bin_value = np.sqrt(bins[i] * bins[i + 1])
        res = compute_theta(bin_pos, bin_pred_pos, plot=False)
        resolutions.append(res)
        bin_medians.append(bin_value)

    state_of_the_art_theta = np.array([0.129, 0.148, 0.120, 0.097, 0.083, 0.082, 0.077, 0.068, 0.061, 0.059, 0.055])
    state_of_the_art_energy = np.array([95, 150, 230, 378, 599, 949, 1504, 2383, 3777, 5986, 9487])

    plt.figure()
    plt.semilogx(10 ** np.array(bin_medians), resolutions, '-o')
    plt.semilogx(state_of_the_art_energy, state_of_the_art_theta, '--*')

    plt.xlabel('Energy')
    plt.ylabel('Angular Resolution')
    plt.title('Angular Resolution of ' + net_name)
    plt.legend([net_name, 'State of the art'])
    plt.grid()
    plt.savefig(fig_folder + '/angular_resolution' + net_name + '.png')
    plt.savefig(fig_folder + '/angular_resolution' + net_name + '.eps')
    plt.show()


plot_angular_resolution(position_te_limato, position_prediction, energy_te_limato, net_name=net_name)
