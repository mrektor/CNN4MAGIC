import matplotlib.pyplot as plt


def plot_interp(npy_event, plot_position=False):
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    ph_m1 = axs[0, 0].imshow(npy_event[1, :, :], origin='lower')
    # if plot_position:
    #     axs[0, 0].plot(res['pos_interp1'][idx][0], res['pos_interp1'][idx][1], 'xr', markersize=18)
    axs[0, 0].set_title('M1, Phe')
    # fig.colorbar(ph_m1, axs[0, 0])

    axs[0, 1].imshow(npy_event[0, :, :], origin='lower')
    # if plot_position:
    #     axs[0, 1].plot(res['pos_interp1'][idx][0], res['pos_interp1'][idx][1], 'xr', markersize=18)
    axs[0, 1].set_title('M1, Time')

    axs[1, 0].imshow(npy_event[2, :, :], origin='lower')
    # if plot_position:
    #     axs[1, 0].plot(res['pos_interp2'][idx][0], res['pos_interp2'][idx][1], 'xr', markersize=18)
    axs[1, 0].set_title('M2, Phe')

    axs[1, 1].imshow(npy_event[3, :, :], origin='lower')
    axs[1, 1].set_title('M2, Time')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig.suptitle('Event ' + str(idx) + ' Energy = ' + str(res['energy'][idx]))
    plt.savefig(f'{folder_pic}/inerp_evt_test_{idx}.png')
    plt.show()
