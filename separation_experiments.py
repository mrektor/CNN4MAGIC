import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import MobileNetV2_separation
from CNN4MAGIC.Generator.training_util import snapshot_training

BATCH_SIZE = 512
machine = '24cores'

# Load the data
train_gn, val_gn = load_generators_diffuse_point(
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
model = MobileNetV2_separation(alpha=1, include_time=False)
# model.load_weights(
#     '/data/new_magic/output_data/snapshots/MobileNetV2_separation_clean6_3punto5_Gold_2019-02-26_13-22-24-Best.h5')
net_name = 'MobileNetV2_separation_10_5'
#%%

result = snapshot_training(model=model,
                           train_gn=train_gn, val_gn=val_gn,
                           net_name=net_name,
                           machine=machine,
                           max_lr=0.05,
                           epochs=7,
                           snapshot_number=5,
                           task='separation',
                           do_telegram=True
                           )
