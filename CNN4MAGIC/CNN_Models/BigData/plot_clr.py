import matplotlib.pyplot as plt
from keras.optimizers import SGD

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR

plt.style.use('seaborn-white')

# Constants
NUM_SAMPLES = 2000
NUM_EPOCHS = 100
BATCH_SIZE = 500
MAX_LR = 0.1

# Data
m1_tr, m2_tr, energy_tr = load_data_append('train', prune=True)

# Model
model = single_DenseNet()
net_name = 'single_DenseNet_vanilla'
clr_triangular = OneCycleLR(NUM_SAMPLES, NUM_EPOCHS, BATCH_SIZE, MAX_LR,
                            end_percentage=0.2, scale_percentage=0.2)

model.compile(optimizer=SGD(0.1), loss='binary_crossentropy', metrics=['accuracy'])

model.fit({'m1': m1_tr, 'm2': m2_tr}, energy_tr, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[clr_triangular],
          verbose=0)

print("LR Range : ", min(clr_triangular.history['lr']), max(clr_triangular.history['lr']))
print("Momentum Range : ", min(clr_triangular.history['momentum']), max(clr_triangular.history['momentum']))

plt.xlabel('Training Iterations')
plt.ylabel('Learning Rate')
plt.title("CLR")
plt.plot(clr_triangular.history['lr'])
plt.savefig(net_name + '_lr.png')
plt.savefig(net_name + '_lr.eps')
plt.show()

plt.xlabel('Training Iterations')
plt.ylabel('Momentum')
plt.title("CLR")
plt.plot(clr_triangular.history['momentum'])
plt.savefig(net_name + '_mom.png')
plt.savefig(net_name + '_mom.eps')
plt.show()
