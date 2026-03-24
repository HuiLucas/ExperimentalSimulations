import matplotlib.pyplot as plt
import numpy as np
from data.datareader import loaded_data
from boundary_corrections.boundary_correction_calculations import BoundaryCorrections


# ---------------------------
# CLEAN FUNCTION
# ---------------------------
def clean_real(arr):
    arr = np.asarray(arr)
    arr = np.real(arr)
    arr[~np.isfinite(arr)] = np.nan
    return arr


# ---------------------------
# LOAD + FILTER DATA
# ---------------------------
data = loaded_data('data/normal_config.mat', 'normal_config')

dE_target = 10.0   # choose desired elevator deflection (degrees)
dE_tolerance = 0.5  # +/- range
data = data.filter(test_point_id__ne='zero_extra') \
                .filter(test_point_id__ne='NoiseBG1') \
                .filter(test_point_id__ne='NoiseBG2') \
                .filter(test_point_id__ne='NoiseBG3') \
                .filter(test_point_id__ne='NoisePO1') \
                .filter(test_point_id__ne='NoisePO2') \
                .filter(test_point_id__ne='NoisePO3') \
                .filter(wind_condition='windOn').filter(
                dE__ge=dE_target - dE_tolerance,
                dE__le=dE_target + dE_tolerance
            )


# ---------------------------
# EXTRACT RAW DATA
# ---------------------------
AoA_unc = clean_real(data['AoA'].values)
CL_unc = clean_real(data['CFZ'].values)
CD_unc = clean_real(data['CFX'].values)
CM_c4_unc = clean_real(data['CMpitch25c'].values)
V_unc = clean_real(data['V'].values)
q_unc = clean_real(data['q'].values)
rho_unc = clean_real(data['rho'].values)

# ---------------------------
# CORRECTION
# ---------------------------
bc = BoundaryCorrections(
    CD_0=0.07,
    V_unc=V_unc,
    rho=rho_unc,
    q_unc=q_unc,
    T=np.zeros_like(AoA_unc),
    alpha_unc=AoA_unc,
    CL_unc=CL_unc,
    CD_unc=CD_unc,
    CM_c4_unc=CM_c4_unc,
    CL_alpha=0.111463,
    test_point_ids=data['test_point_id'].values
)

AoA_cor, V_cor, q_cor, CL_cor, CD_cor, CM_cor = bc.apply_boundary_corrections()

AoA_cor = clean_real(AoA_cor)
CL_cor = clean_real(CL_cor)


# ---------------------------
# SORT FOR CLEAN PLOTTING
# ---------------------------
idx_unc = np.argsort(AoA_unc)
idx_cor = np.argsort(AoA_cor)


# ---------------------------
# PLOT CL-ALPHA
# ---------------------------
plt.figure(figsize=(8, 6))

plt.plot(AoA_unc[idx_unc], CL_unc[idx_unc], 'o', label='Uncorrected')
plt.plot(AoA_cor[idx_cor], CL_cor[idx_cor], 'o', label='Corrected')

plt.xlabel('Angle of Attack (deg)')
plt.ylabel('$C_L$')
plt.title('CL–α Curve: Uncorrected vs Corrected')

plt.grid()
plt.legend()

plt.savefig("results3/CL_alpha_comparison.png")
plt.close()


# ---------------------------
# CD vs AoA
# ---------------------------
plt.figure(figsize=(8, 6))

plt.plot(AoA_unc[idx_unc], clean_real(data['CFX'].values)[idx_unc], 'o', label='CD Uncorrected')
plt.plot(AoA_cor[idx_cor], CD_cor[idx_cor], 'o', label='CD Corrected')

plt.xlabel('Angle of Attack (deg)')
plt.ylabel('$C_D$')
plt.title('CD–α Curve: Uncorrected vs Corrected')
plt.grid()
plt.legend()

plt.savefig("results3/CD_alpha_comparison.png")
plt.close()


# ---------------------------
# CM vs AoA
# ---------------------------
plt.figure(figsize=(8, 6))

plt.plot(AoA_unc[idx_unc], CM_c4_unc[idx_unc], 'o', label='CM Uncorrected')
plt.plot(AoA_cor[idx_cor], CM_cor[idx_cor], 'o', label='CM Corrected')

plt.xlabel('Angle of Attack (deg)')
plt.ylabel('$C_{m,c/4}$')
plt.title('CM–α Curve: Uncorrected vs Corrected')
plt.grid()
plt.legend()

plt.savefig("results3/CM_alpha_comparison.png")
plt.close()


# ---------------------------
# Velocity vs AoA
# ---------------------------
plt.figure(figsize=(8, 6))

plt.plot(AoA_unc[idx_unc], V_unc[idx_unc], 'o', label='V Uncorrected')
plt.plot(AoA_cor[idx_cor], V_cor[idx_cor], 'o', label='V Corrected')

plt.xlabel('Angle of Attack (deg)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs AoA')
plt.grid()

plt.savefig("results3/velocity_alpha.png")
plt.close()
