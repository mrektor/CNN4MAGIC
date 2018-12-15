from keras.models import load_model

from CNN4MAGIC.CNN_Models.SeparationStereo.utils import load_separation_data, plot_confusion_matrix, plot_gammaness

net_name = 'single_DenseNet_25_3_doubleDense'

path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/checkpoints/' + net_name + '.hdf5'
print('Loading model ' + net_name + '...')
model = load_model(path)
m1_te, m2_te, y_true = load_separation_data('test')

print('Making Predictions...')
y_pred = model.predict({'m1': m1_te, 'm2': m2_te})

print('Plotting gammaness...')

plot_gammaness(y_true, y_true, net_name=net_name)
print('Plotting confusion matrix...')

plot_confusion_matrix(y_pred, y_true, ['Hadrons', 'Gammas'], net_name=net_name)

print('All done')
