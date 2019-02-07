from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

BATCH_SIZE = 10
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(batch_size=BATCH_SIZE, want_energy=True)

# %%
a = train_gn[0]
# %%
a[1]
