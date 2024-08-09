import numpy as np
import matplotlib.pyplot as plt

def aaai_fid_compare_vrg():
    """AAAI2025 paper"""
    fig_size = (10, 8)
    tick_size = 25
    legend_size = 22
    xy_label_size = 30
    title_size = 35
    fig_dir = "./checkpoints/2024-07-13_vrg_schedule_trajectory/fig_aaai2025_vrg"

    def _plot():
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', labelsize=tick_size)
        ax.plot(steps, fid_old, linestyle='-', color='c', marker='o', label="SD")  # ab_order=1
        ax.plot(steps, fid_vrg, linestyle='-', color='r', marker='s', label="SD with VRG")
        ax.legend(fontsize=legend_size, loc='upper right')
        ax.set_xlabel('step count', fontsize=xy_label_size)
        ax.set_ylabel('FID      ', fontsize=xy_label_size, rotation=0)
        ax.set_title(r"Stable Diffusion V1", fontsize=title_size)
        f_path = f"{fig_dir}/fid_sd_vrg_steps{steps[0]}-{steps[-1]}.png"
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

    fid_old_all = [327.35, 141.95, 30.02, 16.37, 10.92, 7.43, 5.42, 4.03, 3.07, 1.35]  # ab_order=1, original trajectory
    fid_vrg_all = [298.18,  59.87, 24.53, 15.08, 10.25, 7.05, 5.19, 3.74, 2.85, 1.31]  # ab_order=1, new trajectory
    steps_all = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '15']
    k = 5
    fid_old = fid_old_all[1:k]
    fid_vrg = fid_vrg_all[1:k]
    steps = steps_all[1:k]
    _plot()

    fid_old = fid_old_all[k:]
    fid_vrg = fid_vrg_all[k:]
    steps = steps_all[k:]
    _plot()

if __name__ == '__main__':
    """ entry point """
    aaai_fid_compare_vrg()
