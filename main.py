from datareader import loaded_data
import matplotlib.pyplot as plt
import numpy as np

# Example:

# Load the data
data_normal_configuration = loaded_data('normal_config.mat', 'normal_config')
print(data_normal_configuration) # print the data
print(np.shape(data_normal_configuration)) # perform numpy operations
data_tailoff = loaded_data('tailoff.mat', 'tailoff')
data_propoff = loaded_data('propoff.mat', 'propoff')
data_modeloff = loaded_data('modeloff.mat', 'modeloff')

# Example filtering:
# Data at elevator deflection greater than or equal to 10 degrees
elev_ge_10 = data_normal_configuration.filter(elevator_deflection__ge=10, wind_condition='windOn')  # Use 'windOff' for wind-off data
print('Available data fields:', [f'{name}: {elev_ge_10.datarr.explanations.get(name, 'No description')}' for name in elev_ge_10.datarr.dtype.names])

# Access values
aoa = elev_ge_10['AoA'].values
aos = elev_ge_10['AoS'].values
CL = elev_ge_10['CL'].values
CD = elev_ge_10['CD'].values
CYaw = elev_ge_10['CYaw'].values
CMroll = elev_ge_10['CMroll'].values
CMpitch = elev_ge_10['CMpitch'].values
CMpitch25c = elev_ge_10['CMpitch25c'].values
CMyaw = elev_ge_10['CMyaw'].values
rho = elev_ge_10['rho'].values
V = elev_ge_10['V'].values
pInf = elev_ge_10['pInf'].values
q = elev_ge_10['q'].values
T = elev_ge_10['temp'].values
nu = elev_ge_10['nu'].values
Re = elev_ge_10['Re'].values
J1 = elev_ge_10['J_M1'].values
J2 = elev_ge_10['J_M2'].values
nrotor1 = elev_ge_10['rpsM1'].values
nrotor2 = elev_ge_10['rpsM2'].values
current_motor1 = elev_ge_10['iM1'].values
current_motor2 = elev_ge_10['iM2'].values
temp_motor1 = elev_ge_10['tM1'].values
temp_motor2 = elev_ge_10['tM2'].values
voltage_motor1 = elev_ge_10['vM1'].values
voltage_motor2 = elev_ge_10['vM2'].values
b = elev_ge_10['b'].values
S = elev_ge_10['S'].values
c = elev_ge_10['c'].values
de = elev_ge_10['elevator_deflection'].values

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