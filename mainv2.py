import matplotlib.pyplot as plt
import numpy as np
from data.datareader import loaded_data
from boundary_corrections.boundary_correction_calculations import BoundaryCorrections

if __name__ == '__main__':
    # --- Load data ---
    data_normal_configuration = loaded_data('data/normal_config.mat', 'normal_config')

    # --- Plot configuration ---
    list_of_plots = {
        'CL_AoA_corrections': {
            'x_uncor': 'AoA_uncor',
            'y_uncor': 'CL_uncor',
            'x_cor': 'AoA_cor',
            'y_cor': 'CL_cor',
            'x_label': 'Angle of Attack (deg)',
            'y_label': r'$C_L$',
            'title': r'$C_L$ vs AoA'
        }
    }

    # --- Choose elevator setting ---
    dE = 0.0

    # --- Filter data ---
    elev_chosen = data_normal_configuration.filter(
    dE__ge=dE - 0.5,
    dE__le=dE + 0.5,
    wind_condition='windOn'
    ).filter(
        test_point_id__ne='NoiseBG1'
    ).filter(
        test_point_id__ne='NoiseBG2'
    ).filter(
        test_point_id__ne='NoiseBG3'
    ).filter(
        test_point_id__ne='NoisePO1'
    ).filter(
        test_point_id__ne='NoisePO2'
    ).filter(
        test_point_id__ne='NoisePO3'
    ).filter(
        test_point_id__ne='Noise1'
    ).filter(
        test_point_id__ne='Noise2'
    ).filter(
        test_point_id__ne='Noise3'
    )

    # --- Raw data ---
    AoA_uncor = elev_chosen['AoA'].values
    CL_uncor = elev_chosen['CFZ'].values
    print(elev_chosen)
    V_uncor = elev_chosen['V'].values
    rho = elev_chosen['rho'].values
    Temp = elev_chosen['T'].values
    q_uncor = elev_chosen['q'].values
    CD_uncor = elev_chosen['CFX'].values
    CM_uncor = elev_chosen['CMpitch25c'].values

    # --- Constants ---
    CD_0 = 0.07
    CL_alpha = 0.11106

    # --- Boundary correction ---
    bc = BoundaryCorrections(
        CD_0=CD_0,
        V_unc=V_uncor,
        rho=rho,
        q_unc=q_uncor,
        T=Temp,
        alpha_unc=AoA_uncor,
        CL_unc=CL_uncor,
        CD_unc=CD_uncor,
        CM_c4_unc=CM_uncor,
        CL_alpha=CL_alpha,
        test_point_ids=elev_chosen['test_point_id'].values,
    )

    AoA_cor, _, _, CL_cor, _, _ = bc.apply_boundary_corrections()

    # --- Store variables safely ---
    variables = {
        'AoA_uncor': AoA_uncor,
        'CL_uncor': CL_uncor,
        'AoA_cor': AoA_cor,
        'CL_cor': CL_cor
    }
    
    print(variables['AoA_uncor'])
    print(variables['CL_uncor'])

    # --- Plot loop ---
    for plotting, config in list_of_plots.items():

        # Sort data (kept as requested)
        sort_idx_uncor = np.argsort(variables[config['x_uncor']])
        sort_idx_cor = np.argsort(variables[config['x_cor']])

        x_uncor = variables[config['x_uncor']][sort_idx_uncor]
        y_uncor = variables[config['y_uncor']][sort_idx_uncor]

        x_cor = variables[config['x_cor']][sort_idx_cor]
        y_cor = variables[config['y_cor']][sort_idx_cor]

        # --- Plot (scatter only, no lines) ---
        plt.figure(figsize=(8, 5))

        plt.scatter(x_uncor, y_uncor, label='Uncorrected', alpha=0.7)
        plt.scatter(x_cor, y_cor, label='Corrected', alpha=0.7)

        plt.xlabel(config['x_label'])
        plt.ylabel(config['y_label'])
        plt.title(f"{config['title']} (dE = {dE})")

        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"results2/{plotting}.png")
        plt.close()

    print("Plot(s) saved successfully.")