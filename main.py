

import scipy.io as spio
data = spio.loadmat('data_corrected_by_matlab.mat')

import numpy as np
def add_field(a, name, dtype, value):
    b = np.zeros(a.shape, a.dtype.descr + [(name, dtype)])
    b[list(a.dtype.names)] = a
    b[name] = value
    return b

windOn = data['BAL']['windOn'][0][0]
windOff = data['BAL']['windOff'][0][0]
config = data['BAL']['config'][0][0]

data = {'elev0': add_field(windOn[0][0]['G31_d0'], 'elevator_deflection',
                            np.float64, 0),
        'elev_n10': add_field(windOn[0][0]['G31_de_n10'], 'elevator_deflection',
                            np.float64, -10),
        'elev_20': add_field(windOn[0][0]['G31_de_20'], 'elevator_deflection',
                            np.float64, 20),
        'elev_10': add_field(windOn[0][0]['G31_den10'], 'elevator_deflection',
                            np.float64, 10),
        'elev_n20': add_field(windOn[0][0]['G31_de_n20'] , 'elevator_deflection',
                            np.float64, -20),
        }

# Example, aoa of data at elevator deflection of 20 degrees
elev_20 = data['elev_20']
aoa = elev_20['AoA']
print(aoa)