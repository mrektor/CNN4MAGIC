import pickle

def pickle_read(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def pickle_dump(filepath, object):
    with open(filepath, 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)

#%% Load Energy and position

pos_in_mm = True

# sp_gm will be the argument of the function
sp_gm = '/data4T/CNN4MAGIC/results/MC_classification/experiments/EfficientNet_B0_dropout08/computed_data/crab_separation_EfficientNet_B0_dropout08.pkl'
separation_gamma = pickle_read(sp_gm)

energy_gamma_filename = '/data4T/qualcosadiquellocheeranellahomeemariott/deepmagic/output_data/reconstructions/energy_transfer-SE-inc-v3-snap-LR_0_05HIGH_Best.pkl'
energy_gamma = pickle_read(energy_gamma_filename)

pos_gamma_filename = '/data4T/qualcosadiquellocheeranellahomeemariott/deepmagic/output_data/reconstructions/SE-121-Position-l2-fromepoch80_2019-03-17_23-13-18.pkl'
position_gamm = pickle_read(pos_gamma_filename)
# %

energy_hadrons = read_pkl(
    '/home/emariott/software_magic/output_data/reconstructions/crab_energy_transfer-SE-inc-v3-snap-lowLR_SWA.pkl')
gammaness_hadrons = read_pkl(
    '/home/emariott/software_magic/output_data/reconstructions/crab_separation_MobileNetV2_separation_10_5_notime_alpha1.pkl')
position_hadrons = read_pkl(
    '/home/emariott/software_magic/output_data/reconstructions/position_prediction_crab_SEDenseNet121_position_l2_fromEpoch41_best.pkl')
big_df_crab, evt_list_crab = read_pkl(
    '/home/emariott/magic_data/crab/crab_position_dataframe/big_df_complement_position_interpolated_nan.pkl')

energy_sim = read_pkl('/home/emariott/software_magic/output_data/for_sensitivity/energy_sim_test_point.pkl')
