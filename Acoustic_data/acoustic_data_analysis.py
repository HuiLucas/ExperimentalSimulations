import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.datareader import loaded_acoustic_phase_analysis_data, loaded_acoustic_spectrum_data, loaded_data
import matplotlib.pyplot as plt
import numpy as np

data_normal_configuration = loaded_data('data/normal_config.mat', 'normal_config')
data_tailoff = loaded_data('data/tailoff.mat', 'tailoff')
data_propoff = loaded_data('data/propoff.mat', 'propoff')
data_modeloff = loaded_data('data/modeloff.mat', 'modeloff')

acoustic_spectrum_data_normal_configuration = loaded_acoustic_spectrum_data('data/spectrum_analysis_normal_configuration.mat', 'normal_config', data_normal_configuration)
acoustic_spectrum_data_propoff = loaded_acoustic_spectrum_data('data/spectrum_analysis_propoff.mat', 'propoff', data_propoff)
phase_analysis_normal_configuration = loaded_acoustic_phase_analysis_data('data/acoustic_propeller_phase_analysis_normal_condition.mat', 'normal_config', data_normal_configuration)
acfilt = acoustic_spectrum_data_normal_configuration.filter(AoA__eq=12.)
frequencies = acfilt['flab'].values[:acfilt['N'].values[0][0][0]]
SPSL = acfilt['SPSL'].values[:acfilt['N'].values[0][0][0]]
plt.figure()
plt.scatter([freqplot.squeeze() for freqplot in frequencies], [SPSLplot.squeeze() for SPSLplot in SPSL], s=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('SPL (dB)')
plt.semilogx()
plt.title('Acoustic Spectrum at AoA = 12 degrees')
plt.grid()
yavg = phase_analysis_normal_configuration.filter(AoA__eq=12.)['yAvg'].values
phIntp = phase_analysis_normal_configuration.phIntp

plt.figure()
plt.plot(np.array([np.repeat(phIntpplot.squeeze().reshape( 1, -1), len(yavg), axis=0) for phIntpplot in phIntp]).reshape(-1,1), np.array([yavgplot.squeeze() for yavgplot in yavg]).reshape(-1,1), ms=1)
plt.ylabel('yavg (m)')
plt.xlabel('phIntp (rad)')
plt.title('Phase Analysis at AoA = 12 degrees')
plt.grid()

plt.show()