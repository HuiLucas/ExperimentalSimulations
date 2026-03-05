from datareader import loaded_data
import matplotlib.pyplot as plt
import numpy as np

# Example:

# Load the data
data_normal_configuration = loaded_data('normal_config.mat', 'normal_config')
print(data_normal_configuration) # print the data
print(np.shape(data_normal_configuration)) # perform numpy operations
print(np.shape(data_normal_configuration['AoA']))

#modify data:
#data_normal_configuration['AoA'] =  1.0  # This will set all AoA values to 1 degree
# data_normal_configuration['AoA'] = data_normal_configuration['AoA'].values + 1.0  # This will add 1 degree to all AoA values

# Load the other datasets
data_tailoff = loaded_data('tailoff.mat', 'tailoff')
data_propoff = loaded_data('propoff.mat', 'propoff')
data_modeloff = loaded_data('modeloff.mat', 'modeloff')

# Example filtering:
# Data at elevator deflection 9.5 and 10.5 degrees at wind-on condition, and remove test point 46 (if needed, this is just an example)
elev_approx_10 = data_normal_configuration.filter(dE__ge=9.5, dE__le=10.5, test_point_id__ne='46', wind_condition='windOn')  # Use 'windOff' for wind-off data

# Access values
print('Available data fields:', [f'{name}: {elev_approx_10.explanations.get(name, 'No description')}' for name in elev_approx_10.datarr.dtype.names])
aoa = elev_approx_10['AoA'].values
aos = elev_approx_10['AoS'].values
CL = elev_approx_10['CL'].values
CD = elev_approx_10['CD'].values
CYaw = elev_approx_10['CYaw'].values
CMroll = elev_approx_10['CMroll'].values
CMpitch = elev_approx_10['CMpitch'].values
CMpitch25c = elev_approx_10['CMpitch25c'].values
CMyaw = elev_approx_10['CMyaw'].values
rho = elev_approx_10['rho'].values
V = elev_approx_10['V'].values
pInf = elev_approx_10['pInf'].values
q = elev_approx_10['q'].values
T = elev_approx_10['temp'].values
nu = elev_approx_10['nu'].values
Re = elev_approx_10['Re'].values
J1 = elev_approx_10['J_M1'].values
J2 = elev_approx_10['J_M2'].values
nrotor1 = elev_approx_10['rpsM1'].values
nrotor2 = elev_approx_10['rpsM2'].values
current_motor1 = elev_approx_10['iM1'].values
current_motor2 = elev_approx_10['iM2'].values
temp_motor1 = elev_approx_10['tM1'].values
temp_motor2 = elev_approx_10['tM2'].values
voltage_motor1 = elev_approx_10['vM1'].values
voltage_motor2 = elev_approx_10['vM2'].values
b = elev_approx_10['b'].values
S = elev_approx_10['S'].values
c = elev_approx_10['c'].values
de = elev_approx_10['dE'].values

# Plotting CL, CD, and the drag polar (warning, all kinds of thrust settings are mixed, does not give sensible results)
fig, ax = plt.subplots(1, 3)
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