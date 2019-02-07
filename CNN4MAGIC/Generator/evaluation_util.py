from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error


def evaluate_energy(energy_te, y_pred, net_name):
    energy_te_limato = energy_te[:y_pred.shape[0]]
    print('Start plotting...')
    plot_hist2D(energy_te_limato,
                y_pred,
                net_name,
                fig_folder='CNN4MAGIC/output_data/pictures/energy_reconstruction',
                num_bins=100)

    plot_gaussian_error(energy_te_limato,
                        y_pred,
                        net_name=net_name,
                        fig_folder='CNN4MAGIC/output_data/pictures/energy_reconstruction')
    print('Plotting done.')
