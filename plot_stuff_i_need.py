import matplotlib.pyplot as plt
import pandas as pd

# %%
fig_folder = '/data/new_magic/output_data/pictures/trainings'
df_lowlr = pd.read_csv('/data/new_magic/output_data/csv_logs/transfer-SE-inc-v3-snap_2019-03-19_10-57-34.csv')
df_highlr = pd.read_csv(
    '/data/new_magic/output_data/csv_logs/transfer-SE-inc-v3-snap-LR_0_05HIGH_2019-03-20_01-50-12.csv')

# %%

plt.figure()
plt.plot(df_lowlr['epoch'], df_lowlr['loss'])
plt.plot(df_lowlr['epoch'], df_lowlr['val_loss'])
# plt.grid(linestyle='--')
plt.legend(['Training Loss', 'Validation Loss'])
plt.title('Training of TSE (low LR)')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.savefig(f'{fig_folder}/low_lr.png')
plt.close()

# %%

plt.figure()
plt.plot(df_highlr['epoch'], df_highlr['loss'])
plt.plot(df_highlr['epoch'], df_highlr['val_loss'])
# plt.grid(linestyle='--')
plt.legend(['Training Loss', 'Validation Loss'])
plt.title('Training of TSE (low LR)')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.savefig(f'{fig_folder}/highlr.png')
plt.close()

# %%
plt.figure()
plt.plot(df_lowlr['epoch'], df_lowlr['loss'], marker='o')
plt.plot(df_lowlr['epoch'], df_lowlr['val_loss'], linestyle='--', marker='o')
plt.plot(df_highlr['epoch'][:-2], df_highlr['loss'][1:-1], marker='s')
plt.plot(df_highlr['epoch'][:-2], df_highlr['val_loss'][1:-1], linestyle='--', marker='s')
plt.hlines(0.014417254, 0, 9, linestyles='-.', color='tab:purple')
plt.hlines(0.017092455, 0, 9, linestyles='-.')

plt.grid(which='both', linestyle='--')
plt.legend(
    ['Training Loss (low LR)',
     'Validation Loss (low LR)',
     'Training Loss (high LR)',
     'Validation Loss (high LR)',
     'Test Loss (SWA low LR)',
     'Test Loss (SWA high LR)'])
plt.title('Training of TSE with different LR (high=0.05, low=0.004)')
plt.xlabel('Epoch')
plt.xticks(range(0, 12, 1), range(1, 0 + 12, 1))
plt.xlim([-1, 10])
# plt.ylim([0.012,0.03])
plt.ylabel('Mean Squared Error')
plt.savefig(f'{fig_folder}/bothLR.pdf')
plt.close()
