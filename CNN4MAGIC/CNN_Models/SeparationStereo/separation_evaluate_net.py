from keras.models import load_model

from CNN4MAGIC.CNN_Models.SeparationStereo.utils import *

net_name = 'MobileNetV2'

# Load the model
path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/checkpoints/' + net_name + '.hdf5'
print('Loading model ' + net_name + '...')
model = load_model(path)

# % Load Hadron first and make prediction on them
m1_te, m2_te, y_true_h = load_hadrons('test')
print('Making Predictions...')
y_pred_h = model.predict({'m1': m1_te, 'm2': m2_te}, verbose=1)
# Look at misclassified examples
# %
plot_misclassified_hadrons(m1_te, m2_te, y_pred_h, net_name=net_name)

# %
del m1_te, m2_te  # Free the memory
gc.collect()
#
# Load Gammas
m1_te, m2_te, y_true_g = load_gammas('test')

print('Making Predictions...')
y_pred_g = model.predict({'m1': m1_te, 'm2': m2_te}, verbose=1)

#
plot_misclassified_gammas(m1_te, m2_te, y_pred_g, net_name=net_name)

#
del m1_te, m2_te
gc.collect()

# %%
# Organize better
y_true = np.vstack((y_true_h, y_true_g))
y_pred = np.vstack((y_pred_h, y_pred_g))

# %%
hadrons = y_pred[y_true == 0]
gammas = y_pred[y_true == 1]
# sns.set()
bins = 85
plt.figure()
plt.hist(hadrons, bins=bins, log=True, histtype='step', fill=True, alpha=0.5)
plt.hist(gammas, bins=bins, log=True, histtype='step', fill=True, alpha=0.5)
plt.xlim([0, 1])
plt.legend(['Hadrons', 'Gammas'])
plt.title(net_name)
plt.xlabel('Gammaness')
plt.savefig('/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/pics/gammaness_' + net_name + '.png')
plt.savefig('/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/pics/gammaness_' + net_name + '.eps')
plt.show()

# %% TODO
# tot_gammas = np.sum(y_true)
# tot_hadrons = y_true.shape[0]-tot_gammas
#
# gammaness=np.linspace(0, 0.3, 100)
# epsilon_gamma = []
# epsilon_hadron = []
# Q_abelardo = []
# Q_ruben = []
# significance = []
# num_hadr_gammaness = []
# num_gamma_gammaness = []
#
# for threshold in gammaness:
#     hadr_tmp = np.zeros(y_pred.shape[0])
#     gamma_tmp = np.zeros(y_pred.shape[0])
#     hadr_tmp[y_pred.flatten()<threshold] = 1
#     gamma_tmp[y_pred.flatten()>=threshold] = 1
#     num_hadr = np.sum(hadr_tmp)
#     num_gammas = np.sum(gamma_tmp)
#     num_hadr_gammaness.append(num_hadr)
#     num_gamma_gammaness.append(num_gammas)
#
#     #Efficiency
#     epsilon_gamma.append(num_gammas/tot_gammas)
#     epsilon_hadron.append(num_hadr/tot_hadrons)
#
#     # Q
#     Q_abelardo.append( (num_gammas/tot_gammas)/np.sqrt(num_hadr/tot_hadrons) )
#     Q_ruben.append(num_gammas/np.sqrt(num_hadr))
#
#     # S
#     significance.append((num_gammas-num_hadr)/np.sqrt(num_hadr))
#
# #%%
#
# #%%
# plt.hist(num_gamma_gammaness, log=True, bins=100)
# #%%
# plt.plot(num_gamma_gammaness)
# plt.show()


# %%
# Plot stuff
print('Plotting gammaness...')
plot_gammaness(y_pred, y_true, net_name=net_name)

#%%
print('Plotting confusion matrix...')
plot_confusion_matrix(y_pred, y_true, ['Hadrons', 'Gammas'], net_name=net_name)

print('All done')
