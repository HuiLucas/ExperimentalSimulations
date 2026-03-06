from data.datareader import loaded_data 
from boundary_corrections.boundary_correction_calculations import BoundaryCorrections
import matplotlib.pyplot as plt


if __name__=='__main__':
    data_normal_configuration = loaded_data('data/normal_config.mat', 'normal_config')
    data_tailoff = loaded_data('data/tailoff.mat', 'tailoff')
    data_propoff = loaded_data('data/propoff.mat', 'propoff')
    data_modeloff = loaded_data('data/modeloff.mat', 'modeloff')

    elev_approx_10 = data_normal_configuration.filter(dE__ge=9.5, dE__le=10.5, test_point_id__ne='46', wind_condition='windOn')  # Use 'windOff' for wind-off data
    propoff_basi = data_propoff.filter(dE__ge=9.5, dE__le=10.5,V__ge=29.5,V__le=30.5,dR__ge=-0.5, dR__le=0.5,AoS__ge=-0.0005, AoS__le=0.0005,wind_condition='windOn')  # Use 'windOff' for wind-off data
    print(propoff_basi['AoS'].values)
    bc=BoundaryCorrections(
            CD_0=0.07,  #from prop off data but needs to be refined to make exact TODO lckz huirne realllllllltytytyyy wants to iterate since he is bored
            V_unc=elev_approx_10['V'].values,
            rho=elev_approx_10['rho'].values,
            q_unc=elev_approx_10['q'].values,
            T=elev_approx_10['temp'].values,
            alpha_unc=elev_approx_10['AoA'].values,
            CL_unc=elev_approx_10['CL'].values,
            CD_unc=elev_approx_10['CD'].values,
            CM_c4_unc=elev_approx_10['CMpitch25c'].values,
            CL_alpha=0.11106   #from prop off data but needs to be refined to make exact
            )
    
    alpha_cor, V_cor, q_cor, CL_cor, CD_cor, CM_c4_cor=bc.apply_boundary_corrections()
    aoa = elev_approx_10['AoA'].values
    V = elev_approx_10['V'].values
    q = elev_approx_10['q'].values
    CL = elev_approx_10['CL'].values
    CD = elev_approx_10['CD'].values
    CM_c4 = elev_approx_10['CMpitch25c'].values
    fig=plt.figure()
    ax=fig.subplots(1,4)
    ax[0].scatter(aoa,CL,label='uncorrected')
    ax[0].scatter(alpha_cor,CL_cor,label='corrected')
    ax[0].legend()

    ax[1].scatter(aoa,CD,label='uncorrected')
    ax[1].scatter(alpha_cor,CD_cor,label='corrected')
    ax[1].legend()

    ax[2].scatter(CL,CD,label='uncorrected')
    ax[2].scatter(CL_cor,CD_cor,label='corrected')
    ax[2].legend()

    ax[3].scatter(propoff_basi['AoA'].values,propoff_basi['CL'].values,label='uncorrected')
#     a3[2].scatter(CL_cor,CD_cor,label='corrected')
    ax[3].legend()

    for axs in ax:
        axs.grid()
    plt.show()


    





