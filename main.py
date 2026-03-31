from scipy import interpolate
from data.datareader import loaded_data
from boundary_corrections.boundary_correction_calculations import BoundaryCorrections
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.colors as colors
import matplotlib.cm as cm
from motor_efficiency_calculator import get_motor_efficiency
import pandas as pd
import pingouin as pg
from scipy.optimize import least_squares
from matplotlib.colors import SymLogNorm



if __name__=='__main__':
    data_normal_configuration = loaded_data('data/normal_config.mat', 'normal_config').filter(test_point_id__ne='zero_extra', wind_condition='windOn').filter(test_point_id__ne='NoiseBG1').filter(test_point_id__ne='NoiseBG2').filter(test_point_id__ne='NoiseBG3')
    data_tailoff = loaded_data('data/tailoff.mat', 'tailoff')
    data_propoff = loaded_data('data/propoff.mat', 'propoff').filter(AoS__le=0.0005, AoS__ge=-0.0005)
    data_modeloff = loaded_data('data/modeloff.mat', 'modeloff')


    list_of_plots = {
        'C_P': {'fit': False, 'y_label': r'$C_P$', 'y_data': 'propulsive_power_cor', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'C_P_with_fit': {'fit': 'C_P', 'y_label': r'$C_P$', 'y_data': 'propulsive_power_cor', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'C_P_electric': {'fit': False, 'y_label': r'$C_P$ (electric)', 'y_data': 'prop_electric_power', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'C_P_shaft': {'fit': False, 'y_label': r'$C_P$ (shaft)', 'y_data': 'prop_shaft_power', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'C_D': {'fit': False, 'y_label': r'$C_D$', 'y_data': 'CD_cor', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'C_L': {'fit': False, 'y_label': r'$C_L$', 'y_data': 'CL_cor', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'DC_L':{'fit': False, 'y_label': r'$\Delta C_L = C_L - C_{L,propoff}$', 'y_data': 'DCL_cor', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'DC_L_vs_AoA':{'fit': False, 'y_label': r'$\Delta C_L = C_L - C_{L,propoff}$', 'y_data': 'DCL_cor', 'with_10_mps': False, 'x_data': 'AoA_cor', 'x_label': 'Angle of Attack (degrees)', 'c_data': 'J_cor', 'c_label': 'J (corrected)'},
        'DC_D': {'fit': False, 'y_label': r'$\Delta C_D = C_D - C_{D,propoff}$', 'y_data': 'DCD_cor', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'C_mc4': {'fit': False, 'y_label': r'$C_{m,c/4}$', 'y_data': 'C_m_c4_cor', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'C_T': {'fit': False, 'y_label': r'$C_T$', 'y_data': 'C_T_cor', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'C_T_with_fit': {'fit': 'C_T', 'y_label': r'$C_T$', 'y_data': 'C_T_cor', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'eta_propulsive': {'fit': False, 'y_label': r'$\eta_{propulsive}=\frac{1}{\eta_{recuperation}}$', 'y_data': 'propulsive_efficiency_cor', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'eta_recuperation': {'fit': False, 'y_label': r'$\eta_{recuperation}=\frac{1}{\eta_{propulsive}}$', 'y_data': 'efficiency_recuperation', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'DC_m_c4': {'fit': False, 'y_label': r'$\Delta C_{m,c/4}=C_{m,c/4} - C_{m,c/4,propoff}$', 'y_data': 'DCm_c4_cor', 'with_10_mps': False, 'x_data': 'J_cor', 'x_label': 'J', 'c_data': 'AoA_cor', 'c_label': 'Angle of Attack (degrees)'},
        'CLCD_vs_CL': {'fit': {'ydat':'CL_propoff/CBX_propoff', 'xdat': 'CL_propoff'}, 'y_label': r'$C_L/C_D$', 'y_data': 'CL_cor/CD_cor', 'with_10_mps': False, 'x_data': 'CL_cor', 'x_label': r'$C_L$', 'c_data': 'J_cor', 'c_label': 'J (corrected)'},
        'CL_vs_CD': {'fit': {'ydat':'CL_propoff', 'xdat': 'CBX_propoff'}, 'y_label': r'$C_L$', 'y_data': 'CL_cor', 'with_10_mps': False, 'x_data': 'CD_cor', 'x_label': r'$C_D$', 'c_data': 'J_cor', 'c_label': 'J (corrected)'},
        'Cmc4_vs_AoA': {'fit': {'ydat':'CM_c4_propoff', 'xdat': 'AoA_propoff'}, 'y_label': r'$C_{m,c/4}$', 'y_data': 'C_m_c4_cor', 'with_10_mps': False, 'x_data': 'AoA_cor', 'x_label': r'Angle of Attack (degrees)', 'c_data': 'J_cor', 'c_label': 'J (corrected)'},
        'DCmc4_vs_AoA': {'fit': False, 'y_label': r'$\Delta C_{m,c/4} = C_{m,c/4} - C_{m,c/4,propoff}$ ', 'y_data': 'DCm_c4_cor', 'with_10_mps': False, 'x_data': 'AoA_cor', 'x_label': r'Angle of Attack (degrees)', 'c_data': 'J_cor', 'c_label': 'J (corrected)'},
        'CL_vs_AoA': {'fit': {'ydat':'CL_propoff', 'xdat': 'AoA_propoff'}, 'y_label': r'$C_L$', 'y_data': 'CL_cor', 'with_10_mps': False, 'x_data': 'AoA_cor', 'x_label': r'Angle of Attack (degrees)', 'c_data': 'J_cor', 'c_label': 'J (corrected)'},
    }



    #elev_approx_10 = data_normal_configuration.filter(dE__ge=9.5, dE__le=10.5, test_point_id__ne='46', wind_condition='windOn')  # Use 'windOff' for wind-off data
    #propoff_basi = data_propoff.filter(dE__ge=9.5, dE__le=10.5,V__ge=29.5,V__le=30.5,dR__ge=-0.5, dR__le=0.5,AoS__ge=-0.0005, AoS__le=0.0005,wind_condition='windOn')  # Use 'windOff' for wind-off data
    #print(propoff_basi['AoS'].values)
    #df_pd = data_normal_configuration.to_dataframe()
    dat_correl = {'CL': [], 'DCL': [], 'CD': [], 'DCD': [], 'CT': [], 'Cmc4': [], 'DCmc4': [], 'AoA': [], 'J': [], 'dE': [], 'CL_propoff': [], 'CD_propoff': [], 'CM_c4_propoff': []}
    dE_list = data_normal_configuration['dE'].values.compressed()
    dE_list = dE_list[~np.isnan(dE_list)]
    dE_list = np.unique(dE_list)
    for plotting in list_of_plots.keys(): #list_of_plots.keys()
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots(1, len(dE_list), sharey=True)
        cdat_min = 0
        cdat_max = 0
        scatters = []
        for indd, dE in enumerate(dE_list):
            if not list_of_plots[plotting]['with_10_mps']:
                elev_chosen = data_normal_configuration.filter(dE__ge=dE-0.5, dE__le=dE+0.5 ,V__ge=10.5)
            else:
                elev_chosen = data_normal_configuration.filter(dE__ge=dE-0.5, dE__le=dE+0.5)
            V_uncor = elev_chosen['V'].values


            rho = elev_chosen['rho'].values
            Temp = elev_chosen['temp'].values
            q_uncor = elev_chosen['q'].values
            D = elev_chosen.data_wind_tunnel_test['D']
            S = elev_chosen['S'].values
            b = elev_chosen['b'].values
            c = elev_chosen['c'].values
            J_uncor = V_uncor / (0.5*(elev_chosen['rpsM1'].values + elev_chosen['rpsM2'].values) * D )
            AoA_uncor = elev_chosen['AoA'].values
            CL_uncor = elev_chosen['CFZ'].values
            CBX_uncor = elev_chosen['CFX'].values # balance ref system
            CD_0 = 0.07
            CL_alpha = 0.111463
            #control_power =
            prop_electric_power = (elev_chosen['iM1'].values * elev_chosen['vM1'].values + elev_chosen['iM2'].values * elev_chosen['vM2'].values) / (rho*(0.5*(elev_chosen['rpsM1'].values + elev_chosen['rpsM2'].values))**3 * D**5)
            if plotting == 'C_P_shaft' or plotting == 'C_P_shaft_with_10_mps' or plotting == 'eta_recuperation' or plotting == 'eta_recuperation_with_10_mps' or plotting == 'eta_propulsive' or plotting == 'eta_propulsive_with_10_mps':
                prop_shaft_power = np.array([get_motor_efficiency(0.5*(vol1+vol2), 0.5*(cur1+cur2)) *(vol1 * cur1 + vol2 * cur2) for vol1, cur1, vol2, cur2 in zip(elev_chosen['vM1'].values, elev_chosen['iM1'].values, elev_chosen['vM2'].values, elev_chosen['iM2'].values)]) / (rho*(0.5*(elev_chosen['rpsM1'].values + elev_chosen['rpsM2'].values))**3 * D**5)


            def interpolate_propoff(AoA_angle, dE_value, V_value):
                if V_value >9.5 and V_value< 10.5:
                    print('Warning: V_value is around 10 m/s, which has no propoff corresponding measurement. This is fine for CL and cm but not CD')
                    return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

                filt_propoff = data_propoff.filter(dE__ge=dE_value-0.5, dE__le=dE_value+0.5, V__ge=V_value-1, V__le=V_value+1)
                # interpolate for AoA:
                AoA_propoff_uncor = filt_propoff['AoA'].values
                V_propoff_uncor = filt_propoff['V'].values
                CD_propoff_uncor = filt_propoff['CD'].values
                CL_propoff_uncor = filt_propoff['CL'].values
                CM_c4_propoff_uncor = filt_propoff['CMpitch'].values # according to BS in same reference system as normal config, according to matlab that is 25%chord

                propoff_corr = BoundaryCorrections(
                    CD_0 = CD_0,
                    V_unc = V_propoff_uncor,
                    rho = rho[0],
                    q_unc = 0.5 * rho[0] * V_propoff_uncor**2,
                    T = 0, # propoff
                    alpha_unc = AoA_propoff_uncor,
                    CL_unc = CL_propoff_uncor,
                    CD_unc = CD_propoff_uncor,
                    CM_c4_unc = CM_c4_propoff_uncor,
                    CL_alpha = CL_alpha,
                    test_point_ids = [None] * len(AoA_propoff_uncor)
                )

                [AoA_propoff, V_propoff, q_propoff, CL_propoff, CD_propoff, CM_c4_propoff] = propoff_corr.apply_boundary_corrections()
                CBX_propoff = CD_propoff


                if len(AoA_propoff) < 2:
                    print(f"Not enough AoA data points for interpolation at dE={dE_value}, V={V_value}. Returning NaN.")
                    return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                if len(V_propoff) < 2:
                    print(f"Not enough V data points for interpolation at dE={dE_value}, V={V_value}. Returning NaN.")
                    return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                try:
                    AoA_interp_func = interpolate.interp1d(AoA_propoff, AoA_propoff, bounds_error=False, fill_value="extrapolate")
                    V_interp_func = interpolate.interp1d(AoA_propoff, V_propoff, bounds_error=False, fill_value="extrapolate")
                    CBX_interp_func = interpolate.interp1d(AoA_propoff, CBX_propoff, bounds_error=False, fill_value="extrapolate")
                    CBX_propoff_uncor_interp_func = interpolate.interp1d(AoA_propoff, CD_propoff_uncor, bounds_error=False, fill_value="extrapolate")
                    V_propoff_uncor_interp_func = interpolate.interp1d(AoA_propoff, V_propoff_uncor, bounds_error=False, fill_value="extrapolate")
                    CL_propoff_interp_func = interpolate.interp1d(AoA_propoff, CL_propoff, bounds_error=False, fill_value="extrapolate")
                    CM_c4_propoff_interp_func = interpolate.interp1d(AoA_propoff, CM_c4_propoff, bounds_error=False, fill_value="extrapolate")
                    AoA_propoff_point = AoA_interp_func(AoA_angle)
                    V_propoff_point = V_interp_func(AoA_angle)
                    CBX_propoff_point = CBX_interp_func(AoA_angle)
                    CBX_propoff_uncor_point = CBX_propoff_uncor_interp_func(AoA_angle)
                    V_propoff_uncor_point = V_propoff_uncor_interp_func(AoA_angle)
                    CL_propoff_point = CL_propoff_interp_func(AoA_angle)
                    CM_c4_propoff_point = CM_c4_propoff_interp_func(AoA_angle)
                except Exception as e:
                    print(f"Interpolation error at dE={dE_value}, V={V_value}, AoA={AoA_angle}: {e}. Returning NaN.")
                    return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

                return [CBX_propoff_point, V_propoff_point, CBX_propoff_uncor_point, V_propoff_uncor_point, CL_propoff_point, CM_c4_propoff_point , AoA_propoff_point]
            CBX_propoff, V_propoff, CBX_propoff_uncor, V_propoff_uncor, CL_propoff, CM_c4_propoff, AoA_propoff  = [list(x) for x in zip(*[interpolate_propoff(AoA_angle,dE, V_vel) for AoA_angle, V_vel in zip(AoA_uncor, V_uncor)])]
            CBX_propoff = np.array(CBX_propoff)
            V_propoff = np.array(V_propoff)
            CBX_propoff_uncor = np.array(CBX_propoff_uncor)
            V_propoff_uncor = np.array(V_propoff_uncor)
            C_m_c4_uncor = elev_chosen['CMpitch25c'].values

            C_T_uncor = (S/D**2) * (J_uncor**2/(2*np.cos(np.deg2rad(AoA_uncor)))) *(CBX_propoff_uncor*V_propoff_uncor**2/( V_uncor**2) - CBX_uncor)
            propulsive_power_uncor = C_T_uncor * J_uncor
            if plotting == 'C_P_shaft' or plotting == 'C_P_shaft_with_10_mps' or plotting == 'eta_recuperation' or plotting == 'eta_recuperation_with_10_mps' or plotting == 'eta_propulsive' or plotting == 'eta_propulsive_with_10_mps':
                propulsive_efficiency_uncor = propulsive_power_uncor / prop_shaft_power
            CD_uncor = CBX_uncor + (2*np.cos(np.deg2rad(AoA_uncor))/J_uncor**2) * (D**2/S) * C_T_uncor
            test_point_ids = elev_chosen['test_point_id'].values
            bc_initial=BoundaryCorrections(
                    CD_0=CD_0,
                    V_unc=V_uncor,
                    rho=rho,
                    q_unc=q_uncor,
                    T=C_T_uncor * ((0.5*(elev_chosen['rpsM1'].values + elev_chosen['rpsM2'].values))**2 * rho * D**4) ,
                    alpha_unc=AoA_uncor,
                    CL_unc=CL_uncor,
                    CD_unc=CD_uncor,
                    CM_c4_unc=C_m_c4_uncor,
                    CL_alpha=CL_uncor,
                    test_point_ids=test_point_ids,
                    )


            [AoA_cor, V_cor, q_cor, CL_cor, CD_cor, C_m_c4_cor] = bc_initial.apply_boundary_corrections()
            J_cor = J_uncor * (V_cor / V_uncor)
            exec(f"plotcdata = %s" % list_of_plots[plotting]['c_data'])
            plotcdata = np.real(plotcdata)
            if cdat_max < np.max(plotcdata):
                cdat_max = np.max(plotcdata)
            if cdat_min > np.min(plotcdata):
                cdat_min = np.min(plotcdata)
            DCm_c4_cor = C_m_c4_cor - CM_c4_propoff
            DCL_cor = CL_cor - CL_propoff
            DCD_cor = CD_cor - CBX_propoff
            C_T_cor = C_T_uncor * (V_uncor**2)/(V_cor**2) #(J_cor**2/(2*np.cos(np.deg2rad(AoA_cor)))) * (S/D**2) * (CD_cor - CBX_uncor*((V_uncor**2)/(V_cor**2)))
            propulsive_power_cor = C_T_cor * J_cor
            if plotting == 'C_P_shaft' or plotting == 'C_P_shaft_with_10_mps' or plotting == 'eta_recuperation' or plotting == 'eta_recuperation_with_10_mps' or plotting == 'eta_propulsive' or plotting == 'eta_propulsive_with_10_mps':
                propulsive_efficiency_cor = propulsive_power_cor / prop_shaft_power
            if plotting == 'C_P_shaft' or plotting == 'C_P_shaft_with_10_mps' or plotting == 'eta_recuperation' or plotting == 'eta_recuperation_with_10_mps' or plotting == 'eta_propulsive' or plotting == 'eta_propulsive_with_10_mps':
                efficiency_recuperation = 1./propulsive_efficiency_cor
            exec(f"plotydata = %s" % list_of_plots[plotting]['y_data'])
            exec(f"plotxdata = %s" % list_of_plots[plotting]['x_data'])
            scatters.append(ax[indd].scatter(plotxdata,plotydata ,c=plotcdata))
            # plot CT = −0.0051J4 + 0.0959J3 − 0.5888J2 + 1.0065J − 0.1353
            J_linspace = np.linspace(np.min(J_cor), np.max(J_cor), 100)
            # plot CP = −0.0093𝐽4 + 0.1832𝐽3 − 1.1784𝐽2 + 2.2005𝐽 − 0.5180

            if type(list_of_plots[plotting]['fit']) == dict:
                xdat = np.array(eval(list_of_plots[plotting]['fit']['xdat']))
                ydat = np.array(eval(list_of_plots[plotting]['fit']['ydat']))
                # order xdat and ydat points according to xdat so that the line plot is correct:
                sorted_indices = np.argsort(xdat)
                xdat = xdat[sorted_indices]
                ydat = ydat[sorted_indices]

                ax[indd].plot(xdat, ydat, label='Prop-Off', ls='-', marker='o', markersize=1)
                ax[indd].legend()
            elif list_of_plots[plotting]['fit'] == 'C_P':
                ax[indd].plot(J_linspace, -0.0093*J_linspace**4 + 0.1832*J_linspace**3 - 1.1784*J_linspace**2 + 2.2005*J_linspace - 0.5180, label='CP fit', ls='-', marker='o', markersize=1)
            elif list_of_plots[plotting]['fit'] == 'C_T':
                ax[indd].plot(J_linspace, -0.0051*J_linspace**4 + 0.0959*J_linspace**3 - 0.5888*J_linspace**2 + 1.0065*J_linspace - 0.1353, label='fit', ls='-', marker='o', markersize=1)

            ax[indd].set_title(f"dE = {dE} degrees")
            ax[indd].set_xlabel(f"{list_of_plots[plotting]['x_label']} (cor)")
            fig.supylabel(f"{list_of_plots[plotting]['y_label']} (cor)")
            ax[indd].grid()
            print(indd)
            dat_correl['CL'].extend(CL_cor.reshape(-1))
            dat_correl['DCL'].extend(DCL_cor.reshape(-1))
            dat_correl['CD'].extend(CD_cor.reshape(-1))
            dat_correl['DCD'].extend(DCD_cor.reshape(-1))
            dat_correl['CT'].extend(C_T_cor.reshape(-1))
            dat_correl['Cmc4'].extend(C_m_c4_cor.reshape(-1))
            dat_correl['DCmc4'].extend(DCm_c4_cor.reshape(-1))
            dat_correl['AoA'].extend(AoA_cor)
            dat_correl['J'].extend(J_cor.reshape(-1))
            dat_correl['dE'].extend([dE] * len(J_cor.reshape(-1)))
            dat_correl['CL_propoff'].extend(np.array(CL_propoff).reshape(-1))
            dat_correl['CD_propoff'].extend(np.array(CBX_propoff).reshape(-1))
            dat_correl['CM_c4_propoff'].extend(np.array(CM_c4_propoff).reshape(-1))





            # # corresponding prop off data for the same test points as the normal configuration data, to be used for comparison
            # prop_off_comparision_list = {'V': [], 'AoA': [], 'rho': [], 'q': [], 'temp': [], 'CL': [], 'CD': [], 'CM_c4': [], 'test_point_id': []}
            # for ind, test_point_id in enumerate(bc_initial.test_point_ids):
            #     similar_propoff_data = data_propoff.filter(V__le=bc_initial.V_unc[ind] + 0.5,
            #                                                 V__ge=bc_initial.V_unc[ind] - 0.5,
            #                                                 AoS=0,
            #                                                dE__le  = dE+0.5,
            #                                                dE__ge  = dE-0.5,
            #                                                AoA__le = bc_initial.alpha_unc[ind] + 0.1,
            #                                                AoA__ge = bc_initial.alpha_unc[ind] - 0.1,
            #                                                )
            #     prop_off_comparision_list['V'].append(similar_propoff_data['V'].values[0])
            #     prop_off_comparision_list['AoA'].append(similar_propoff_data['AoA'].values[0])
            #     prop_off_comparision_list['rho'].append(similar_propoff_data['rho'].values[0])
            #     prop_off_comparision_list['q'].append(similar_propoff_data['q'].values[0])
            #     prop_off_comparision_list['temp'].append(similar_propoff_data['temp'].values[0])
            #     prop_off_comparision_list['CL'].append(similar_propoff_data['CL'].values[0])
            #     prop_off_comparision_list['CD'].append(similar_propoff_data['CD'].values[0])
            #     prop_off_comparision_list['CM_c4'].append(similar_propoff_data['CMpitch25c'].values[0])
            #     prop_off_comparision_list['test_point_id'].append(similar_propoff_data['test_point_id'].values[0])
            # bc_propoff = BoundaryCorrections(
            #         CD_0=0.07,  #from prop off data but needs to be refined to make exact TODO lckz huirne realllllllltytytyyy wants to iterate since he is bored
            #         V_unc=prop_off_comparision_list['V'],
            #         rho=prop_off_comparision_list['rho'],
            #         q_unc=prop_off_comparision_list['q'],
            #         T=prop_off_comparision_list['temp'],
            #         alpha_unc=prop_off_comparision_list['AoA'],
            #         CL_unc=prop_off_comparision_list['CL'],
            #         CD_unc=prop_off_comparision_list['CD'],
            #         CM_c4_unc=prop_off_comparision_list['CM_c4'],
            #         CL_alpha=0.11106,  #from prop off data but needs to be refined to make exact,
            #         test_point_ids=prop_off_comparision_list['test_point_id'],
            #         )
        df_pd = pd.DataFrame(dat_correl)[['CD', 'AoA', 'J', 'dE']]
        print(df_pd.pcorr())
        cmap = plt.cm.viridis
        norm = colors.Normalize(vmin=cdat_min, vmax=cdat_max)
        for scc in scatters:
            scc.set_cmap(cmap)
            scc.set_norm(norm)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, pad=0.15,orientation='horizontal', location='bottom')
        cbar.set_label(f"{list_of_plots[plotting]['c_label']}")
        #fig.tight_layout()
        plt.savefig(f"results2/{plotting}.png")
        plt.close(fig)


    def aerodynamic_residuals(params, data):
        (b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, alpha_L0, alpha_M0, kk, CL_alpha_propoff, alpha_L0_propoff, CL_delta_propoff, Cm_c4_alpha_propoff, alpha_M0_propoff, Cm_c4_delta_propoff, CD_0_propoff) = params

        # Unpack the measured flight/wind-tunnel data
        J = np.asarray(data['J'], dtype=float)
        alpha = np.asarray(data['AoA'], dtype=float)
        delta_e = np.asarray(data['dE'], dtype=float)

        CT_meas = np.asarray(data['CT'], dtype=float)
        CL_meas = np.asarray(data['CL'], dtype=float)
        CD_meas = np.asarray(data['CD'], dtype=float)
        Cm_meas = np.asarray(data['Cmc4'], dtype=float)

        CL_propoff = np.asarray(data['CL_propoff'], dtype=float)
        Cm_propoff = np.asarray(data['CM_c4_propoff'], dtype=float)
        CD_propoff = np.asarray(data['CD_propoff'], dtype=float)



        # Evaluate the mathematical model using current parameter guesses
        CT_pred = b0 + b1 * J

        CL_pred = CL_propoff + b2 * alpha + b3 * J * alpha - b3 * alpha_L0 * J - b2 * alpha_L0

        CD_pred = (CD_propoff - kk* CL_propoff ** 2) + b4 + b5 * J + (b6 + b7 * J) * CL_meas ** 2

        Cm_pred = Cm_propoff + b10 * alpha + b11 * J * alpha - b11 * alpha_M0 * J + \
                  b8 * delta_e + b9 * J * delta_e - b10 * alpha_M0

        CL_propoff_pred = CL_alpha_propoff * (alpha - alpha_L0_propoff) + CL_delta_propoff * delta_e
        Cm_propoff_pred = Cm_c4_alpha_propoff * (alpha - alpha_M0_propoff) + Cm_c4_delta_propoff * delta_e
        CD_propoff_pred = CD_0_propoff + CL_propoff_pred**2 * kk

        # Calculate residuals (Difference between measured data and model prediction)
        res_CT = CT_meas - CT_pred
        res_CL = CL_meas - CL_pred
        res_CD = CD_meas - CD_pred
        res_Cm = Cm_meas - Cm_pred
        res_CL_propoff = CL_propoff - CL_propoff_pred
        res_Cm_propoff = Cm_propoff - Cm_propoff_pred
        res_CD_propoff = CD_propoff - CD_propoff_pred

        # Optional: You can apply weighting here if CD is much smaller than CL, e.g.:
        # res_CD = res_CD * 10

        # Combine all residuals into a single 1D array for the solver
        return np.concatenate((res_CT, res_CL, res_CD, res_Cm, res_CL_propoff, res_Cm_propoff, res_CD_propoff))

    initial_guess = np.zeros(22)

    # Run the optimization
    # Using the Levenberg-Marquardt algorithm ('lm') which is standard for NLLS
    result = least_squares(
        aerodynamic_residuals,
        initial_guess,
        args=(dat_correl,),
        method='lm'
    )

    param_names = [f"Beta_{i}" for i in range(12)] + ["Alpha_L=0", "Alpha_M=0", "k", "CL_alpha_propoff", "Alpha_L0_propoff", "CL_delta_propoff", "Cm_c4_alpha_propoff", "Alpha_M0_propoff", "Cm_c4_delta_propoff", "CD_0_propoff"]

    print("--- Optimized Parameters ---")
    for name, value in zip(param_names, result.x):
        print(f"{name:<10}: {value:>10.5f}")

    # Trimmed condition analysis
    def delta_e_for_trim(alpha, J, params):
        ddd = ( params[18] * (alpha - params[19])   + params[10] * alpha + params[11] * J * alpha - params[11] * params[13] * J  +  - params[10] * params[13])/ (-(params[20] + params[9] * J + params[8]) )
        return ddd

    def trim_drag_lift(alpha, J, params):
        delta_e = delta_e_for_trim(alpha, J, params)
        CL_propoff = params[15] * (alpha - params[16]) + params[17] * delta_e
        CL = CL_propoff + params[2] * alpha + params[3] * J * alpha - params[3] * params[12] * J - params[2] * params[12]
        CD = params[21] + params[4] + params[5] * J + (params[6] + params[7] * J) * CL ** 2
        return CD, CL

    def delta_e_for_trim_propoff(alpha, params):
        ddd = -params[18] * (alpha - params[19]) / params[20]
        return ddd
    # one figure with all three plots dE vs aoa with color J, CL vs CD with color J and CL vs AoA with color J
    alpha_linspace = np.linspace(-5, 14, 100)
    J_linspace = np.linspace(0, 8, 100)
    alpha_grid, J_grid = np.meshgrid(alpha_linspace, J_linspace)
    CD_grid, CL_grid = trim_drag_lift(alpha_grid, J_grid, result.x)
    fig = plt.figure(figsize=(15, 5))
    ax = fig.subplots(1, 3)
    contour1 = ax[0].contourf(alpha_grid, J_grid, CD_grid, levels=50, cmap='viridis')
    ax[0].contour(alpha_grid, J_grid, CD_grid, levels=20, colors='k', linewidths=0.5)
    fig.colorbar(contour1, ax=ax[0], label='CD at Trim')
    ax[0].set_title('CD at Trim Condition')
    ax[0].set_xlabel('Angle of Attack (degrees)')
    ax[0].set_ylabel('Advance Ratio (J)')
    contour2 = ax[1].contourf(alpha_grid, J_grid, CL_grid, levels=50, cmap='viridis')
    ax[1].contour(alpha_grid, J_grid, CL_grid, levels=20, colors='k', linewidths=0.5)
    fig.colorbar(contour2, ax=ax[1], label='CL at Trim')
    ax[1].set_title('CL at Trim Condition')
    ax[1].set_xlabel('Angle of Attack (degrees)')
    ax[1].set_ylabel('Advance Ratio (J)')
    Z = CL_grid/CD_grid
    ax[2].contourf(alpha_grid, J_grid, Z, levels=50, cmap='viridis')
    ax[2].contour(alpha_grid, J_grid, Z, levels=20, colors='k', linewidths=0.5)
    fig.colorbar(contour2, ax=ax[2], label='CL/CD at Trim')
    ax[2].set_title('CL/CD at Trim Condition')
    ax[2].set_xlabel('Angle of Attack (degrees)')
    ax[2].set_ylabel('Advance Ratio (J)')
    plt.tight_layout()
    plt.savefig("results2/trim_conditions.png")
    plt.close()
    # delta_e vs alpha for different J values
    fig = plt.figure(figsize=(10, 5))
    ax = fig.subplots(1, 1)
    delta_e_propoff = delta_e_for_trim_propoff(alpha_linspace, result.x)
    for J_val in [0, 4, 8]:
        delta_e_grid = delta_e_for_trim(alpha_linspace, J_val, result.x)
        ax.plot(alpha_linspace, delta_e_grid, label=f'J={J_val}')
    ax.plot(alpha_linspace, delta_e_propoff, label=f'Prop-Off', ls='--', color='k')
    ax.set_title('Elevator Deflection for Trim Condition')
    ax.set_xlabel('Angle of Attack (degrees)')
    ax.set_ylabel('Elevator Deflection (degrees)')
    ax.legend()
    plt.tight_layout()
    plt.savefig("results2/delta_e_trim.png")
    plt.close()



#     alpha_cor, V_cor, q_cor, CL_cor, CD_cor, CM_c4_cor=bc.apply_boundary_corrections()
#     aoa = elev_approx_10['AoA'].values
#     V = elev_approx_10['V'].values
#     q = elev_approx_10['q'].values
#     CL = elev_approx_10['CL'].values
#     CD = elev_approx_10['CD'].values
#     CM_c4 = elev_approx_10['CMpitch25c'].values
#     fig=plt.figure()
#     ax=fig.subplots(1,4)
#     ax[0].scatter(aoa,CL,label='uncorrected')
#     ax[0].scatter(alpha_cor,CL_cor,label='corrected')
#     ax[0].legend()
#
#     ax[1].scatter(aoa,CD,label='uncorrected')
#     ax[1].scatter(alpha_cor,CD_cor,label='corrected')
#     ax[1].legend()
#
#     ax[2].scatter(CL,CD,label='uncorrected')
#     ax[2].scatter(CL_cor,CD_cor,label='corrected')
#     ax[2].legend()
#
#     ax[3].scatter(propoff_basi['AoA'].values,propoff_basi['CL'].values,label='uncorrected')
# #     a3[2].scatter(CL_cor,CD_cor,label='corrected')
#     ax[3].legend()
#
#     for axs in ax:
#         axs.grid()
#     plt.show()


    





