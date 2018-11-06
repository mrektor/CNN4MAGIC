import pickle

import matplotlib.pyplot as plt

# %%

with open('/data2T/mariotti_data_2/energy_MC_diffuse/result_za05to35_8_821327_Y.pkl', 'rb') as f:
    dict = pickle.load(f)

# %%
print(dict.keys())
# %%
dict['energy'][2]
# %%
for idx in range(len(dict['energy'])):
    plt.figure()
    plt.suptitle('Energy: ' + str(dict['energy'][idx]))
    plt.subplot(1, 2, 1)
    plt.imshow(dict['M1_interp'][idx])
    plt.title('M1, position: ' + str(dict['positionM1'][idx]))

    plt.subplot(1, 2, 2)
    plt.imshow(dict['M2_interp'][idx])
    plt.title('M2, position: ' + str(dict['positionM2'][idx]))

    plt.tight_layout()

    plt.savefig('/data/mariotti_data/data_process/sanity_checks_pic/check_' + str(idx) + '.png')
