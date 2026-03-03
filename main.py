from datareader import loaded_data
import matplotlib.pyplot as plt

# Example:

# Load the data
data_normal_configuration = loaded_data('normal_config.mat', 'normal_config')
print(data_normal_configuration)
data_tailoff = loaded_data('tailoff.mat', 'tailoff')


# Data at elevator deflection of 20 degrees
elev_20 = data_normal_configuration.filter(elevator_deflection=20, wind_condition='windOn')  # Use 'windOff' for wind-off data
print('Available data fields:', [f'{name}: {elev_20.datarr.explanations.get(name, 'No description')}' for name in elev_20.datarr.dtype.names])

# Access values
aoa = elev_20['AoA'].values
aos = elev_20['AoS'].values
CL = elev_20['CL'].values
CD = elev_20['CD'].values
CYaw = elev_20['CYaw'].values
CMroll = elev_20['CMroll'].values
CMpitch = elev_20['CMpitch'].values
CMpitch25c = elev_20['CMpitch25c'].values
CMyaw = elev_20['CMyaw'].values
rho = elev_20['rho'].values
V = elev_20['V'].values
pInf = elev_20['pInf'].values
q = elev_20['q'].values
T = elev_20['temp'].values
nu = elev_20['nu'].values
Re = elev_20['Re'].values
J1 = elev_20['J_M1'].values
J2 = elev_20['J_M2'].values
nrotor1 = elev_20['rpsM1'].values
nrotor2 = elev_20['rpsM2'].values
current_motor1 = elev_20['iM1'].values
current_motor2 = elev_20['iM2'].values
temp_motor1 = elev_20['tM1'].values
temp_motor2 = elev_20['tM2'].values
voltage_motor1 = elev_20['vM1'].values
voltage_motor2 = elev_20['vM2'].values
b = elev_20['b'].values
S = elev_20['S'].values
c = elev_20['c'].values
de = elev_20['elevator_deflection'].values

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