import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import dummy_cnn, MobileNetV2_separation
from CNN4MAGIC.Generator.training_util import snapshot_training

BATCH_SIZE = 512
machine = '24cores'

# Load the datatr
train_gn, val_gn, test = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=False,
    want_label=True,
    machine=machine,
    clean=True,
    include_time=False)

# Load the model
print('Loading the Neural Network...')
# model = MobileNetV2_separation(alpha=0.2, include_time=False)

# %%
model = dummy_cnn()
# model.load_weights(
#     '/data/new_magic/output_data/snapshots/SimplicioNet_2019-03-20_14-15-34-7.h5')
net_name = 'SimplicioNet'

#%%

result = snapshot_training(model=model,
                           train_gn=train_gn, val_gn=val_gn,
                           net_name=net_name,
                           machine=machine,
                           max_lr=0.01,
                           epochs=6,
                           snapshot_number=6,
                           swa=3,
                           task='separation',
                           do_telegram=True
                           )
