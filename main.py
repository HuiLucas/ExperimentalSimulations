from data.datareader import loaded_data 
from boundary_corrections.boundary_correction_calculations import BoundaryCorrections

if __name__=='__main__':
    data_normal_configuration = loaded_data('data/normal_config.mat', 'normal_config')
    data_tailoff = loaded_data('data/tailoff.mat', 'tailoff')
    data_propoff = loaded_data('data/propoff.mat', 'propoff')
    data_modeloff = loaded_data('data/modeloff.mat', 'modeloff')

    elev_approx_10 = data_normal_configuration.filter(dE__ge=9.5, dE__le=10.5, test_point_id__ne='46', wind_condition='windOn')  # Use 'windOff' for wind-off data
    bc=BoundaryCorrections(
            CD_0=0.07,
            V_unc=elev_approx_10['V'].values,
            rho=elev_approx_10['rho'].values,
            q_unc=elev_approx_10['q'].values,
            T=elev_approx_10['temp'].values,
            alpha_unc=elev_approx_10['AoA'].values,
            CL_unc=elev_approx_10['CL'].values,
            CD_unc=elev_approx_10['CD'].values,
            CM_c4_unc=elev_approx_10['CMpitch25c'].values,
            )
    aoa = elev_approx_10['AoA'].values



