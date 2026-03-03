import matplotlib.pyplot as plt
import scipy.io as spio
data_wind_tunnel_test = spio.loadmat('tailon.mat')
data_tailoff = spio.loadmat('tailoff.mat')

import numpy as np
def add_field(a, name, dtype, value):
    b = np.zeros(a.shape, a.dtype.descr + [(name, dtype)])
    b[list(a.dtype.names)] = a
    b[name] = value
    return b

windOn = data_wind_tunnel_test['BAL']['windOn'][0][0]
windOff = data_wind_tunnel_test['BAL']['windOff'][0][0]
config = data_wind_tunnel_test['BAL']['config'][0][0]

data_normal_configuration = {'elev0': {'windOn': add_field(windOn[0][0]['G31_d0'], 'elevator_deflection',
                                                np.float64, 0), 'windOff': add_field(windOff[0][0]['G31_d0'], 'elevator_deflection'
                                       , np.float64, 0)},
        'elev_n10': {'windOn': add_field(windOn[0][0]['G31_de_n10'], 'elevator_deflection',
                            np.float64, -10) , 'windOff': add_field(windOff[0][0]['G31_de_n10'], 'elevator_deflection',
                                       np.float64, -10)},
        'elev_20': {'windOn': add_field(windOn[0][0]['G31_de_20'], 'elevator_deflection',
                            np.float64, 20) , 'windOff': add_field(windOff[0][0]['G31_de_20'], 'elevator_deflection',
                                       np.float64, 20)},
        'elev_10': {'windOn': add_field(windOn[0][0]['G31_den10'], 'elevator_deflection',
                            np.float64, 10) , 'windOff': add_field(windOff[0][0]['G31_den10'], 'elevator_deflection',
                                       np.float64, 10)},
        'elev_n20': {'windOn': add_field(windOn[0][0]['G31_de_n20'] , 'elevator_deflection',
                            np.float64, -20) , 'windOff': add_field(windOff[0][0]['G31_de_n20'], 'elevator_deflection',
                                       np.float64, -20)}
                             }
tailOff_windOn = data_tailoff['BAL']['windOn'][0][0]['tailOff_beta0_balance']
tailOff_windOff = data_tailoff['BAL']['windOff'][0][0]['tailOff_beta0_balance']
data_tailoff = {'windOn': tailOff_windOn, 'windOff': tailOff_windOff}

# Example, data at elevator deflection of 20 degrees
elev_20 = data_normal_configuration['elev_20']['windOn']  # Use 'windOff' for wind-off data
aoa = elev_20['AoA'][0][0].squeeze()
aos = elev_20['AoS'][0][0].squeeze()
CL = elev_20['CL'][0][0].squeeze()
CD = elev_20['CD'][0][0].squeeze()
CYaw = elev_20['CYaw'][0][0].squeeze()
CMroll = elev_20['CMroll'][0][0].squeeze()
CMpitch = elev_20['CMpitch'][0][0].squeeze()
CMpitch25c = elev_20['CMpitch25c'][0][0].squeeze()
CMyaw = elev_20['CMyaw'][0][0].squeeze()
rho = elev_20['rho'][0][0].squeeze() # Corrected
V = elev_20['V'][0][0].squeeze()  # Corrected
pInf = elev_20['pInf'][0][0].squeeze()  # Corrected
q = elev_20['q'][0][0].squeeze()  # Corrected
T = elev_20['temp'][0][0].squeeze()
nu = elev_20['nu'][0][0].squeeze() # Corrected
Re = elev_20['Re'][0][0].squeeze() # Corrected
J1 = elev_20['J_M1'][0][0].squeeze()
J2 = elev_20['J_M2'][0][0].squeeze()
nrotor1 = elev_20['rpsM1'][0][0].squeeze()
nrotor2 = elev_20['rpsM2'][0][0].squeeze()
current_motor1 = elev_20['iM1'][0][0].squeeze()
current_motor2 = elev_20['iM2'][0][0].squeeze()
temp_motor1 = elev_20['tM1'][0][0].squeeze()
temp_motor2 = elev_20['tM2'][0][0].squeeze()
voltage_motor1 = elev_20['vM1'][0][0].squeeze()
voltage_motor2 = elev_20['vM2'][0][0].squeeze()
b = elev_20['b'][0][0].squeeze()
S = elev_20['S'][0][0].squeeze()
c = elev_20['c'][0][0].squeeze()
de = elev_20['elevator_deflection'][0][0].squeeze()
print('Available data fields:', elev_20.dtype.names)
fig, ax = plt.subplots(1,3)
ax[0].scatter(aoa, CL, label='CL')
ax[0].set_xlabel('Angle of Attack (degrees)')
ax[0].set_ylabel('Lift Coefficient (CL)')
ax[0].legend()
ax[1].scatter(aoa, CD, label='CD', color='orange')
ax[1].set_xlabel('Angle of Attack (degrees)')
ax[1].set_ylabel('Drag Coefficient (CD)')
ax[1].legend()
ax[2].scatter(CL, CD, label='Drag Polar', color='green')
ax[2].set_xlabel('Lift Coefficient (CL)')
ax[2].set_ylabel('Drag Coefficient (CD)')
ax[2].legend()
plt.tight_layout()
plt.show()
