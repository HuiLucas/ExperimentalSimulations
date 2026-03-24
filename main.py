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


if __name__=='__main__':
    data_normal_configuration = loaded_data('data/normal_config.mat', 'normal_config').filter(test_point_id__ne='zero_extra')
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
    corrected_data = {}
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
                elev_chosen = data_normal_configuration.filter(dE__ge=dE-0.5, dE__le=dE+0.5, wind_condition='windOn', test_point_id__ne='NoiseBG1' ,V__ge=10.5).filter(test_point_id__ne='NoiseBG2').filter(test_point_id__ne='NoiseBG3')
            else:
                elev_chosen = data_normal_configuration.filter(dE__ge=dE-0.5, dE__le=dE+0.5, test_point_id__ne='NoiseBG1', wind_condition='windOn').filter(test_point_id__ne='NoiseBG2').filter(test_point_id__ne='NoiseBG3')
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

            C_T_uncor = (S/D**2) * (J_uncor**2/(2*np.cos(np.deg2rad(AoA_uncor)))) *(CBX_propoff_uncor*V_propoff_uncor**2/( V_uncor**2) - CBX_uncor)  # During iteration change order of operations so that uncorr values are used when needed
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
            C_T_cor = (J_cor**2/(2*np.cos(np.deg2rad(AoA_cor)))) * (S/D**2) * (CD_cor - CBX_uncor*((V_uncor**2)/(V_cor**2)))
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
            ax[indd].set_xlabel(f"{list_of_plots[plotting]['x_label']} (corrected)")
            fig.supylabel(f"{list_of_plots[plotting]['y_label']} (corrected)")
            ax[indd].grid()
            print(indd)





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


    





