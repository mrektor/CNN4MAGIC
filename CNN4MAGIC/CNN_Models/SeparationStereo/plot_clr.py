import matplotlib.pyplot as plt
from keras.optimizers import SGD

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.CNN_Models.BigData.loader import load_data_append
from CNN4MAGIC.CNN_Models.BigData.stereo_models import single_DenseNet

plt.style.use('seaborn-white')

# Constants
NUM_EPOCHS = 3
BATCH_SIZE = 64
MAX_LR = 0.5

# Data
m1_tr, m2_tr, energy_tr = load_data_append('train', prune=True)
NUM_SAMPLES = energy_tr.shape[0]

# Model
model = single_DenseNet()
net_name = 'single_DenseNet_vanilla'
clr_triangular = OneCycleLR(NUM_SAMPLES, NUM_EPOCHS, BATCH_SIZE, MAX_LR,
                            end_percentage=0.2, scale_percentage=0.2)

model.compile(optimizer=SGD(0.1), loss='binary_crossentropy', metrics=['accuracy'])

model.fit({'m1': m1_tr, 'm2': m2_tr}, energy_tr, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[clr_triangular],
          verbose=1)

print("LR Range : ", min(clr_triangular.history['lr']), max(clr_triangular.history['lr']))
print("Momentum Range : ", min(clr_triangular.history['momentum']), max(clr_triangular.history['momentum']))

plt.xlabel('Training Iterations')
plt.ylabel('Learning Rate')
plt.title("CLR")
plt.plot(clr_triangular.history['lr'])
plt.savefig('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/learning_rates_pics/' + net_name + '_lr.png')
plt.savefig('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/learning_rates_pics/' + net_name + '_lr.eps')
plt.show()

plt.xlabel('Training Iterations')
plt.ylabel('Momentum')
plt.title("CLR")
plt.plot(clr_triangular.history['momentum'])
plt.savefig('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/learning_rates_pics' + net_name + '_mom.png')
plt.savefig('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/learning_rates_pics' + net_name + '_mom.eps')
plt.show()
