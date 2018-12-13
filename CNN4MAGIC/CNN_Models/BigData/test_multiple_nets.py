import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from CNN4MAGIC.CNN_Models.BigData.loader import load_data_append
from CNN4MAGIC.CNN_Models.BigData.utils import compute_bin_gaussian_error

models = []
legends = []
m1_te, m2_te, energy_te = load_data_append('test', prune=True)
y_true = np.log10(energy_te)

fig_width = 9
fig, axs = plt.subplots(1, 2, figsize=(fig_width, fig_width * 0.618))

results = {}
for path in models:
    model = load_model(path)
    y_pred = model.predict({'m1': m1_te, 'm2': m2_te})

    bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, y_pred, net_name='', num_bins=10,
                                                                        fig_folder='', plot=False)
    results[path] = [bins_mu, bins_sigma, bins_median_value]
