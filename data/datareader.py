import scipy.io as spio
import numpy as np
import copy
from numpy.lib import recfunctions as rfn
import matplotlib.pyplot as plt
import pandas as pd


FIELD_EXPLANATIONS = {
            'AoA': 'Angle of Attack in degrees',
            'AoS': 'Angle of Sideslip in degrees',
            'CL': 'Lift Coefficient. Not calculated by MATLAB for windoff',
            'CD': 'Drag Coefficient. Not calculated by MATLAB for windoff',
            'CYaw': 'Side Force Coefficient. Not calculated by MATLAB for windoff',
            'CMroll': 'Rolling Moment Coefficient. Not calculated by MATLAB for windoff',
            'CMpitch': 'Pitching Moment Coefficient. Not calculated by MATLAB for windoff',
            'CMpitch25c': 'Pitching Moment Coefficient at 25% chord. Not calculated by MATLAB for windoff, propoff',
            'CMyaw': 'Yawing Moment Coefficient. Not calculated by MATLAB for windoff',
            'rho': 'Air Density in kg/m^3. Corrected from measured value by the Matlab code to account for sensor calibration. Not measured for windoff, propoff condition',
            'V': 'Freestream Velocity in m/s. Corrected from measured value by the Matlab code to account for sensor calibration. Not applicable to windoff condition',
            'pInf': 'Freestream Pressure in Pa. Corrected from measured value by the Matlab code to account for sensor calibration. Not measured for windoff, propoff condition',
            'q': 'Dynamic Pressure in Pa. Corrected from measured value by the Matlab code to account for sensor calibration. Not measured for windoff, propoff condition',
            'temp': 'Air Temperature in K. Not measured for propoff condition',
            'nu': 'Kinematic Viscosity in m^2/s. Calculated from corrected parameters. Not calculated by MATLAB for windoff, propoff',
            'Re': 'Reynolds Number. Corrected from measured value by the Matlab code to account for sensor calibration. Not calculated by MATLAB for windoff, propoff',
            'J_M1': 'Advance Ratio of Motor 1. Not applicable to windoff, propoff condition',
            'J_M2': 'Advance Ratio of Motor 2. Not applicable to windoff, propoff condition',
            'rpsM1': 'Rotations per second of Motor 1. Not applicable to windoff, propoff condition',
            'rpsM2': 'Rotations per second of Motor 2. Not applicable to windoff, propoff condition',
            'iM1': 'Current of Motor 1 in Amperes. Not applicable to windoff, propoff condition',
            'iM2': 'Current of Motor 2 in Amperes. Not applicable to windoff, propoff condition',
            'tM1': 'Temperature of Motor 1 in Celsius. Not applicable to windoff, propoff condition',
            'tM2': 'Temperature of Motor 2 in Celsius. Not applicable to windoff, propoff condition',
            'vM1': 'Voltage of Motor 1 in Volts. Not applicable to windoff, propoff condition',
            'vM2': 'Voltage of Motor 2 in Volts. Not applicable to windoff, propoff condition',
            'b': 'Wing Span in meters. Not calculated by MATLAB for windoff, propoff',
            'S': 'Wing Planform Area in square meters. Not calculated by MATLAB for windoff, propoff',
            'c': 'Mean Aerodynamic Chord in meters. Not calculated by MATLAB for windoff, propoff',
            'dE': 'Elevator Deflection in degrees. Not applicable for tail off.',
            'dR': 'Rudder Deflection in degrees. Not applicable for tail off.',
            'wind_condition': 'Wind condition during the test, either "windOn" or "windOff". Unknown for propoff',
            'pMic': 'Microphone pressure data. For acoustic spectrum data and phase data only, not applicable to aerodynamic data',
            'oneP': 'Pulse signal data for propeller rotation. For acoustic spectrum data and phase data only, not applicable to aerodynamic data',
            'yAvg': 'Phase averaged pressure data. For phase data only, not applicable to aerodynamic data and acoustic spectrum data',
            'nS': 'Number of samples in the acoustic spectrum data. For acoustic spectrum data only, not applicable to aerodynamic data and phase data',
            'tMeas': 'Measurement time in seconds for the acoustic spectrum data. For acoustic spectrum data only, not applicable to aerodynamic data and phase data',
            't': 'Time array for spectrum data. For acoustic spectrum data only, not applicable to aerodynamic data and phase data',
            'Naq': 'Number of acquired samples for the acoustic spectrum data. For acoustic spectrum data only, not applicable to aerodynamic data and phase data',
            'dt': 'Time step for the acoustic spectrum data. For acoustic spectrum data only, not applicable to aerodynamic data and phase data',
            'B': 'Number of frequencies in frequency bins (ensembles) for the acoustic spectrum data. For acoustic spectrum data only, not applicable to aerodynamic data and phase data',
            'df': 'Frequency resolution for the acoustic spectrum data. For acoustic spectrum data only, not applicable to aerodynamic data and phase data',
            'flab': 'Frequency array for the acoustic spectrum data. For acoustic spectrum data only, not applicable to aerodynamic data and phase data',
            'ApOASPL_dB': 'Overall Sound Pressure Level in decibels (avg over time), calculated from the acoustic spectrum data. For acoustic spectrum data only, not applicable to aerodynamic data and phase data',
            'N': 'Fourier analysis ensemble size (number of bins) for the acoustic spectrum data. For acoustic spectrum data only, not applicable to aerodynamic data and phase data',
            'SPSL': 'Sound Pressure Level in decibels for each frequency bin, calculated from the acoustic spectrum data. For acoustic spectrum data only, not applicable to aerodynamic data and phase data',
            'test_point_id': 'Identifier for the test point, e.g. "13", "14", ..., "Noise1", "Noise2", etc. '
        }

def add_field(a, name, dtype, value):
    b = np.zeros(a.shape, a.dtype.descr + [(name, object)])
    b[list(a.dtype.names)] = a
    if np.isscalar(value) or value == None:
        b[name][...] = [[np.full_like(b['AoA'][0][0], np.array([[value]]), dtype=dtype)]]
    else:
        raise NotImplementedError
    return b

def add_field2(a, name, dtype, value):
    b = rfn.append_fields(a, name, [value], dtypes=dtype , usemask=False)
    return b

def align_dtype(arr, target_dtype,targetshape=None):
    if targetshape is None:
        out = np.ma.empty(arr.shape, dtype=target_dtype)
    else:
        out = np.ma.empty(np.shape(targetshape), dtype=target_dtype)
        for field in target_dtype.names:
            if field not in arr.dtype.names:
                if field != 'B' and field != 'B16zeroed':
                    out.data[0][0][field] = np.zeros((np.shape(arr[0][0]['run'])[0] , np.shape(targetshape[0][0][field])[1]))
                    out.mask[0][0][field] = True
                else:
                    out.data[0][0][field] = np.zeros(
                        (np.shape(arr[0][0]['run'])[0], 6))
                    out.mask[0][0][field] = True
            else:
                if np.shape(arr[field][0][0]) == () or np.shape(arr[field][0][0]) == (1, 1):
                    out[0][0][field] = np.full((np.shape(arr[0][0]['run'])[0] , np.shape(targetshape[0][0]['run'])[1]), arr[field][0][0])
                else:
                    out[0][0][field] = arr[field][0][0]

    return out


class loaded_data:
    def __init__(self, data_file, configuration):
        self.configuration = configuration
        self.data_file = data_file
        self.explanations = FIELD_EXPLANATIONS
        if configuration == 'normal_config':
            self.data_wind_tunnel_test = spio.loadmat(data_file)
            windOn = self.data_wind_tunnel_test['BAL']['windOn'][0][0]
            windOff = self.data_wind_tunnel_test['BAL']['windOff'][0][0]
            config = self.data_wind_tunnel_test['BAL']['config'][0][0]
            data_normal_configuration2 = copy.deepcopy(add_field(add_field(windOn[0][0]['G31_d0'], 'dE',
                                                              np.float64, 0), 'wind_condition', 'U10', 'windOn'))
            prototype_dtype = data_normal_configuration2.dtype
            data_normal_configuration = copy.deepcopy(data_normal_configuration2)

            data_normal_configuration_array = [align_dtype(add_field(add_field(windOn[0][0]['G31_d0'], 'dE',
                                                                       np.float64, 0), 'wind_condition', 'U10', 'windOn'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOff[0][0]['G31_d0'], 'dE'
                                                                        , np.float64, np.nan), 'wind_condition', 'U10', 'windOff'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOn[0][0]['G31_de_n10'], 'dE',
                                                                 np.float64, -10), 'wind_condition', 'U10', 'windOn'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOn[0][0]['G31_de_20'], 'dE',
                                                                 np.float64, 20), 'wind_condition', 'U10', 'windOn'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOn[0][0]['G31_den10'], 'dE',
                                                                 np.float64, 10), 'wind_condition', 'U10', 'windOn'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOn[0][0]['G31_de_n20'], 'dE',
                                                                    np.float64, -20), 'wind_condition', 'U10', 'windOn'), prototype_dtype, data_normal_configuration2)]



            for field in prototype_dtype.names:
                arrs_for_concatenation = []
                for indd, arrr in enumerate(data_normal_configuration_array):
                    if type(arrr[field][0][0]) == np.ma.MaskedArray:
                        arrs_for_concatenation.append(arrr[field][0][0])
                    else:
                        arrs_for_concatenation.append(np.ma.masked_array(arrr[field][0][0], mask=np.zeros_like(arrr[field][0][0], dtype=bool)))
                data_normal_configuration[field][0][0] = np.ma.concatenate(arrs_for_concatenation)
            data_normal_configuration = np.ma.array(data_normal_configuration, mask=np.zeros_like(data_normal_configuration, dtype=bool))
            self.datarr = data_normal_configuration

        elif configuration == 'tailoff':
            self.data_wind_tunnel_test = spio.loadmat(data_file)
            tailOff_windOn = self.data_wind_tunnel_test['BAL']['windOn'][0][0]['tailOff_beta0_balance'][0][0]
            tailOff_windOff = self.data_wind_tunnel_test['BAL']['windOff'][0][0]['tailOff_beta0_balance'][0][0]
            data_tailoff2 = copy.deepcopy(add_field(add_field(tailOff_windOn, 'dE',
                                                                       np.float64, None), 'wind_condition', 'U10', 'windOn'))
            prototype_dtype = data_tailoff2.dtype
            data_tailoff = copy.deepcopy(data_tailoff2)

            data_tailoff_array = [align_dtype(data_tailoff2, prototype_dtype, data_tailoff2), align_dtype(add_field(add_field(tailOff_windOff, 'dE'
                                                                            , np.float64, None), 'wind_condition', 'U10', 'windOff'), prototype_dtype, data_tailoff2)]


            for field in prototype_dtype.names:
                arrs_for_concatenation = []
                for indd, arrr in enumerate(data_tailoff_array):
                    if type(arrr[field][0][0]) == np.ma.MaskedArray:
                        arrs_for_concatenation.append(arrr[field][0][0])
                    else:
                        arrs_for_concatenation.append(
                            np.ma.masked_array(arrr[field][0][0], mask=np.zeros_like(arrr[field][0][0], dtype=bool)))
                data_tailoff[field][0][0] = np.ma.concatenate(arrs_for_concatenation)
            data_tailoff = np.ma.array(data_tailoff, mask=np.zeros_like(data_tailoff, dtype=bool))
            self.datarr = data_tailoff
        elif configuration == 'propoff':
            ## TODO: PROPOFF NEEDS TO BE MATLAB-Corrected still!! Ask prof if it needs to be corrected because it is already in coefficient form. Also 4409 data points is bit large
            self.data_wind_tunnel_test = spio.loadmat(data_file)
            propOff_windOn = self.data_wind_tunnel_test['propOff']
            data_propoff = np.ma.masked_array(add_field(propOff_windOn, 'wind_condition', 'U10', 'windOn'))
            self.datarr = data_propoff

        elif configuration == 'modeloff':
            # TODO: check whether modeloff needs to be corrected by MATLAB code, and if so, implement that correction
            self.data_wind_tunnel_test = spio.loadmat(data_file)
            modelOff_windOn = self.data_wind_tunnel_test['modelOff']
            data_modeloff = np.ma.masked_array(add_field(add_field(modelOff_windOn, 'dE',
                                                                       np.float64, None), 'wind_condition', 'U10', 'windOn'))
            self.datarr = data_modeloff
        self.add_test_point_id()

    def add_test_point_id(self):
        if 'test_point_id' not in self.datarr.dtype.names:
            if self.configuration == 'tailoff':
                self.datarr = add_field(self.datarr, 'test_point_id', 'U25', 'tailoff (from Sinnige)')
                return
            elif self.configuration == 'propoff':
                self.datarr = add_field(self.datarr, 'test_point_id', 'U25', 'propoff (from Sinnige)')
                return
            elif self.configuration == 'modeloff':
                self.datarr = add_field(self.datarr, 'test_point_id', 'U25', 'modeloff (from Sinnige)')
                return
            else:
                self.datarr = add_field(self.datarr, 'test_point_id', 'U25', None)
        self.datarr['test_point_id'][0][0][0] =  '13'
        self.datarr['test_point_id'][0][0][1] =  '14'
        self.datarr['test_point_id'][0][0][2] =  '15'
        self.datarr['test_point_id'][0][0][3] =  '16'
        self.datarr['test_point_id'][0][0][4] = '17'
        self.datarr['test_point_id'][0][0][5] = '18'
        self.datarr['test_point_id'][0][0][6] = '19'
        self.datarr['test_point_id'][0][0][7] = 'Noise2'
        self.datarr['test_point_id'][0][0][8] = 'NoisePO2'
        self.datarr['test_point_id'][0][0][9] = 'NoiseBG2'
        self.datarr['test_point_id'][0][0][10] = 'wo3'
        self.datarr['test_point_id'][0][0][11] = 'wo25'
        self.datarr['test_point_id'][0][0][12] = 'wo4'
        self.datarr['test_point_id'][0][0][13] = 'wo11'
        self.datarr['test_point_id'][0][0][14] = 'woextra'
        self.datarr['test_point_id'][0][0][15] = 'wo0'
        self.datarr['test_point_id'][0][0][16] = 'wo5'
        self.datarr['test_point_id'][0][0][17] = 'wo2'
        self.datarr['test_point_id'][0][0][18] = 'wo24'
        self.datarr['test_point_id'][0][0][19] = 'wo1'
        self.datarr['test_point_id'][0][0][20] = '0'
        self.datarr['test_point_id'][0][0][21] = '1'
        self.datarr['test_point_id'][0][0][22] = '2'
        self.datarr['test_point_id'][0][0][23] = '3'
        self.datarr['test_point_id'][0][0][24] = '4'
        self.datarr['test_point_id'][0][0][25] = '5'
        self.datarr['test_point_id'][0][0][26] = '6'
        self.datarr['test_point_id'][0][0][27] = '7'
        self.datarr['test_point_id'][0][0][28] = '8'
        self.datarr['test_point_id'][0][0][29] = '9'
        self.datarr['test_point_id'][0][0][30] = '10'
        self.datarr['test_point_id'][0][0][31] = '11'
        self.datarr['test_point_id'][0][0][32] = '12'
        self.datarr['test_point_id'][0][0][33] = 'Noise1'
        self.datarr['test_point_id'][0][0][34] = 'NoisePO1'
        self.datarr['test_point_id'][0][0][35] = 'NoiseBG1'
        self.datarr['test_point_id'][0][0][36] = '32'
        self.datarr['test_point_id'][0][0][37] = '33'
        self.datarr['test_point_id'][0][0][38] = '34'
        self.datarr['test_point_id'][0][0][39] = '35'
        self.datarr['test_point_id'][0][0][40] = '36'
        self.datarr['test_point_id'][0][0][41] = '37'
        self.datarr['test_point_id'][0][0][42] = '38'
        self.datarr['test_point_id'][0][0][43] = '39'
        self.datarr['test_point_id'][0][0][44] = 'Noise4'
        self.datarr['test_point_id'][0][0][45] = 'NoisePO4'
        self.datarr['test_point_id'][0][0][46] = 'zero_extra'
        self.datarr['wind_condition'][0][0][46] = 'windOff'
        self.datarr['test_point_id'][0][0][47] = '21'
        self.datarr['test_point_id'][0][0][48] = '22'
        self.datarr['test_point_id'][0][0][49] = '23'
        self.datarr['test_point_id'][0][0][50] = '24'
        self.datarr['test_point_id'][0][0][51] = '25'
        self.datarr['test_point_id'][0][0][52] = '26'
        self.datarr['test_point_id'][0][0][53] = '27'
        self.datarr['test_point_id'][0][0][54] = '28'
        self.datarr['test_point_id'][0][0][55] = '29'
        self.datarr['test_point_id'][0][0][56] = '30'
        self.datarr['test_point_id'][0][0][57] = '31'
        self.datarr['test_point_id'][0][0][58] = 'Noise3'
        self.datarr['test_point_id'][0][0][59] = 'NoisePO3'
        self.datarr['test_point_id'][0][0][60] = 'NoiseBG3'
        self.datarr['test_point_id'][0][0][61] = '40'
        self.datarr['test_point_id'][0][0][62] = '41'
        self.datarr['test_point_id'][0][0][63] = '42'
        self.datarr['test_point_id'][0][0][64] = '43'
        self.datarr['test_point_id'][0][0][65] = '44'
        self.datarr['test_point_id'][0][0][66] = '45'
        self.datarr['test_point_id'][0][0][67] = '46'
        self.datarr['test_point_id'][0][0][68] = '47'




    def __getitem__(self, item, acoustic=False):
        #retarr = [test_point[item] for test_point in self.datarr]
        if not acoustic:
            retobj = object.__new__(loaded_data)
        else:
            retobj = object.__new__(loaded_acoustic_spectrum_data)
        retobj.datarr = self.datarr[[item]]
        retobj.explanations = FIELD_EXPLANATIONS
        retobj.data_file = self.data_file
        retobj.data_wind_tunnel_test = self.data_wind_tunnel_test
        return  retobj #np.ma.array(retarr).ravel()

    def filter(self, acoustic=False, **kwargs):
        """
        Filter self.datarr based on field conditions and return masked array.

        Supports multiple filtering options:
        - field=value: exact match for any data type
        - field__eq=value: explicit equality (same as field=value)
        - field__ne=value: not equal
        - field__lt=value: less than
        - field__le=value: less than or equal
        - field__gt=value: greater than
        - field__ge=value: greater than or equal

        Multiple filters are combined with AND logic.

        Parameters
        ----------
        **kwargs : dict
            Filter conditions as keyword arguments

        Returns
        -------
        np.ma.array
            Filtered masked array with True mask where conditions are met

        Examples
        --------
        # Filter by exact value
        filtered = data.filter(wind_condition='windOn', dE=20)

        # Filter with inequalities
        filtered = data.filter(AoA__gt=5, AoA__lt=15)

        # Combined filters
        filtered = data.filter(wind_condition='windOn', Re__gt=1e6)
        """
        if not kwargs:
            # Return all data if no filters specified
            return self.datarr

        # Initialize mask: True means keep the element
        mask = np.ones(self.datarr.shape, dtype=bool)

        for filter_spec, value in kwargs.items():
            # Parse the filter specification
            parts = filter_spec.split('__')
            field_name = parts[0]
            operator = parts[1] if len(parts) > 1 else 'eq'

            # Get field data
            if field_name not in self.datarr.dtype.names:
                raise ValueError(f"Field '{field_name}' not found in data. "
                               f"Available fields: {self.datarr.dtype.names}")

            if not acoustic:
                field_data = self.datarr[field_name][0][0]
            else:
                field_data = self.datarr[field_name]

            # Apply operator
            if operator == 'eq':
                condition = (field_data == value)
            elif operator == 'ne':
                condition = (field_data != value)
            elif operator == 'lt':
                condition = (field_data < value)
            elif operator == 'le':
                condition = (field_data <= value)
            elif operator == 'gt':
                condition = (field_data > value)
            elif operator == 'ge':
                condition = (field_data >= value)
            else:
                raise ValueError(f"Unknown operator '{operator}'. "
                               f"Supported: eq, ne, lt, le, gt, ge")

            # Combine with existing mask using AND logic
            mask = np.array(mask & condition)

        # Return filtered masked array

        if not acoustic:
            filtarr = np.ma.array(copy.deepcopy(self.datarr))
            for field in self.datarr.dtype.names:
                if np.shape(self.datarr[field][0][0])[1] != 1:
                    #maskt = np.repeat(mask, np.shape(self.datarr[field][0][0])[1], axis=1)
                    filtarr[field] = [[self.datarr[field][0][0][mask.squeeze()]]]
                else:
                    filtarr[field] = [[self.datarr[field][0][0][mask].reshape(-1,1)]]
        else:
            filtarr = np.ma.array(copy.deepcopy(self.datarr))[mask.squeeze()]
            # for field in self.datarr.dtype.names:
            #     if np.shape(self.datarr[field])[1] != 1:
            #         #maskt = np.repeat(mask, np.shape(self.datarr[field])[1], axis=1)
            #         filtarr[field] = self.datarr[field][mask.squeeze()]
            #     else:
            #         filtarr[field] = self.datarr[field][mask].reshape(-1,1)
        if not acoustic:
            retobj = object.__new__(loaded_data)
        else:
            retobj = object.__new__(loaded_acoustic_spectrum_data)
        retobj.datarr = filtarr
        retobj.explanations = FIELD_EXPLANATIONS
        retobj.data_file = self.data_file
        retobj.data_wind_tunnel_test = self.data_wind_tunnel_test
        return retobj

    def __setitem__(self, key, value, acoustic=False):
        """
        Set a field in self.datarr with a numpy array, masked array, or scalar value.

        Parameters
        ----------
        key : str
            Field name to set
        value : np.ndarray, np.ma.MaskedArray, or scalar
            - Array of shape (1, len(self.datarr)) or (len(self.datarr),)
              containing the values to assign to the field
            - Scalar value (int, float, str, etc.) to set all entries to that value

        Raises
        ------
        ValueError
            If field doesn't exist or array size doesn't match

        Examples
        --------
        # Set with 1D array
        object_name['AoA'] = aoa_array

        # Set with 2D array (1, n)
        object_name['CL'] = cl_array.reshape(1, -1)

        # Set all entries to a scalar value
        object_name['flag'] = 1
        object_name['temperature'] = 25.5
        """
        # Check if field exists
        if key not in self.datarr.dtype.names:
            raise ValueError(f"Field '{key}' not found in data. "
                           f"Available fields: {self.datarr.dtype.names}")

        # Check if value is a scalar (int, float, str, bool, etc.)
        if np.isscalar(value):
            # Set all entries to the scalar value
            if not acoustic:
                self.datarr[key][0][0] = np.full_like(self.datarr[key][0][0], value)
            else:
                self.datarr[key] = np.full_like(self.datarr[key], value)
            return

        # Convert input to numpy array if needed
        if isinstance(value, np.ma.MaskedArray):
            value_arr = value
        else:
            value_arr = np.asarray(value)

        # Flatten the array to 1D if it's 2D with shape (1, n)
        if value_arr.ndim == 2:
            if value_arr.shape[0] == 1:
                value_arr = value_arr.reshape(-1)
            else:
                if not acoustic:
                    raise ValueError(f"Array shape {value_arr.shape} not compatible. "
                               f"Expected (1, {len(self.datarr[key][0][0])}) or ({len(self.datarr[key][0][0])},)")
                else:
                    raise ValueError(f"Array shape {value_arr.shape} not compatible. "
                               f"Expected (1, {len(self.datarr[key])}) or ({len(self.datarr[key])},)")
        elif value_arr.ndim != 1:
            raise ValueError(f"Array must be 1D or 2D, got {value_arr.ndim}D")

        # Check size matches
        if not acoustic:
            if len(value_arr) != len(self.datarr[key][0][0]):
                raise ValueError(f"Array size {len(value_arr)} doesn't match data size {len(self.datarr[key][0][0])}. "
                               f"Expected size ({len(self.datarr[key][0][0])},) or (1, {len(self.datarr[key][0][0])})")
        else:
            if len(value_arr) != len(self.datarr[key]):
                raise ValueError(f"Array size {len(value_arr)} doesn't match data size {len(self.datarr[key])}. "
                               f"Expected size ({len(self.datarr[key])},) or (1, {len(self.datarr[key])})")

        # Set the field
        if not acoustic:
            self.datarr[key][0][0] = value_arr
        else:
            self.datarr[key] = value_arr

    def __array__(self, acoustic=False):
        returner = []
        if not acoustic:
            for field in self.datarr.dtype.names:
                if np.shape(self.datarr[field][0][0])[1] != 1:
                    for ind in range(np.shape(self.datarr[field][0][0])[1]):
                        returner.append(self.datarr[field][0][0][:, ind].reshape(-1, 1))
                else:
                    returner.append(self.datarr[field][0][0])
        else:
            for field in self.datarr.dtype.names:
                if np.shape(self.datarr[field])[1] != 1:
                    for ind in range(np.shape(self.datarr[field])[1]):
                        returner.append(self.datarr[field][:, ind].reshape(-1, 1))
                else:
                    returner.append(self.datarr[field])
        returner = np.array(returner)
        return returner

    #pd dataframe:
    def to_dataframe(self, acoustic=False):
        if not acoustic:
            data_dict = {field: self.datarr[field][0][0].ravel() for field in self.datarr.dtype.names if np.shape(self.datarr[field][0][0])[1] == 1}
            for field in self.datarr.dtype.names:
                if np.shape(self.datarr[field][0][0])[1] != 1:
                    for ind in range(np.shape(self.datarr[field][0][0])[1]):
                        data_dict[f"{field}_{ind}"] = self.datarr[field][0][0][:, ind].ravel()
        else:
            data_dict = {field: self.datarr[field].ravel() for field in self.datarr.dtype.names if np.shape(self.datarr[field])[1] == 1}
            for field in self.datarr.dtype.names:
                if self.datarr[field].ndim != 1:
                    for ind in range(np.shape(self.datarr[field])[1]):
                        data_dict[f"{field}_{ind}"] = self.datarr[field][:, ind].ravel()
        df = pd.DataFrame(data_dict)
        return df

    def __str__(self):
        fields = self.datarr.dtype.names
        rows_str = []

        for i, row in enumerate(self.datarr):
            rows_str.append(f"Record {i}:")
            for f in fields:
                val = row[f]
                try:
                    explen = self.explanations[f]
                except KeyError:
                    explen = 'No description'
                if np.ma.is_masked(val):
                    val_str = "--"
                else:
                    val_str = str(val)
                rows_str.append(f"{f}\n{explen}\n{val_str}\n")
            rows_str.append(20*"======")  # blank line between records

        return "\n".join(rows_str)

    def __repr__(self):
        if hasattr(self, 'data_file'):
            return f"{self.__class__.__name__}(data_file={self.data_file!r})"
        else:
            return f"{self.__class__.__name__}(configuration=unknown)"

    @property
    def values(self, acoustic=False):
        # Return plain numeric array (for plotting, math, etc.)
        plain = np.ma.column_stack(
            [self.datarr[name] for name in self.datarr.dtype.names]
        )
        out = np.ma.filled(plain, np.nan)
        if not acoustic:
            return out[0][0].ravel()
        else:
            return out.ravel()

class loaded_acoustic_spectrum_data(loaded_data):
    # "This will
    # be done by the lab-exercise supervisor before the lab exercise using a pistonphone (G.R.A.S.
    # Pistonphone 42AA). Post-processing routines will be made available on Brightspace to process the data
    # obtained from the inflow microphone. The calibration constant will be included in these files."
    # TODO: where is this calibration constant, cannot be found in the matlab scripts? Or is it the: 20 micro-Pa reference pressure, [Pa]
    # NOTE: Propeller noise analysis vs phase only performed for prop-on condition (so not BG and prop off)
    def __init__(self, data_file, configuration, associated_aerodynamic_data):
        self.configuration = configuration
        self.data_file = data_file
        self.associated_aerodynamic_data = associated_aerodynamic_data
        if configuration == 'normal_config':
            self.data_wind_tunnel_test = spio.loadmat(data_file)
            acoustic_windOn = self.data_wind_tunnel_test['MIC']
            acoustic_data_normal_configuration2 = copy.deepcopy(
                add_field2(add_field2(acoustic_windOn[:,0][0], 'dE',
                                    np.float64, 20), 'wind_condition', 'U10', 'windOn'))
            prototype_dtype = acoustic_data_normal_configuration2.dtype
            # 0 32, 1 33, 2 34, 3 35, 4 36, 5 37, 6 38, 7 39, 8 40, 9 N1, 10 N2, 11 N3, 12 N4
            # 0 up to and incl 8: elevator deflection of 20 deg, N1: -10, N2: 0, N3: 10, N4: 20
            self.dictionary_of_test_point_id_with_array_index = {
                '32': 0,
                '33': 1,
                '34': 2,
                '35': 3,
                '36': 4,
                '37': 5,
                '38': 6,
                '39': 7,
                '40': 8,
                'Noise1': 9,
                'Noise2': 10,
                'Noise3': 11,
                'Noise4': 12
            }
            acoustic_data_normal_configuration_array = [
                add_field2(add_field2(add_field2(acoustic_windOn[:,0][0], 'dE',
                                                np.float64, 20), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', '32'),
                add_field2(add_field2(add_field2(acoustic_windOn[:,1][0], 'dE',
                                                np.float64, 20), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', '33'),
                add_field2(add_field2(add_field2(acoustic_windOn[:,2][0], 'dE',
                                                np.float64, 20), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', '34'),
                add_field2(add_field2(add_field2(acoustic_windOn[:,3][0], 'dE',
                                                np.float64, 20), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', '35'),
                add_field2(add_field2(add_field2(acoustic_windOn[:,4][0], 'dE',
                                                np.float64, 20), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', '36'),
                add_field2(add_field2(add_field2(acoustic_windOn[:,5][0], 'dE',
                                                np.float64, 20), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', '37'),
                add_field2(add_field2(add_field2(acoustic_windOn[:,6][0], 'dE',
                                                np.float64, 20), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', '38'),
                add_field2(add_field2(add_field2(acoustic_windOn[:,7][0], 'dE',
                                                np.float64, 20), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', '39'),
                add_field2(add_field2(add_field2(acoustic_windOn[:,8][0], 'dE',
                                                np.float64, 20), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', '40'),
                add_field2(add_field2(add_field2(acoustic_windOn[:,9][0], 'dE',
                                                np.float64, -10), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', 'Noise1'),
                add_field2(add_field2(add_field2(acoustic_windOn[:,10][0], 'dE',
                                                np.float64, 0), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', 'Noise2'),
                add_field2(add_field2(add_field2(acoustic_windOn[:,11][0], 'dE',
                                                np.float64, 10), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', 'Noise3'),
                add_field2(add_field2(add_field2(acoustic_windOn[:,12][0], 'dE',
                                                np.float64, 20), 'wind_condition', 'U10', 'windOn'), 'test_point_id', 'U10', 'Noise4'),
                ]
            arrays = [[row] for row in acoustic_data_normal_configuration_array]
            acoustic_data_normal_configuration = np.ma.array(arrays)
            self.datarr = acoustic_data_normal_configuration
            self.explanations = FIELD_EXPLANATIONS
            self.fill_in_missing_values()
        elif configuration == 'propoff':
            self.data_wind_tunnel_test = spio.loadmat(data_file)
            acoustic_propOff_windOn = self.data_wind_tunnel_test['MIC']


            acoustic_data_normal_propoff_configuration_array = [
                add_field2(add_field2(add_field2(acoustic_propOff_windOn[:, 0][0], 'dE',
                                                 np.float64, 0), 'wind_condition', 'U10', 'windOn'), 'test_point_id',
                           'U25', 'propoff (from Sinnige)'),
            ]
            arrays = [row[0] for row in acoustic_data_normal_propoff_configuration_array]
            acoustic_data_normal_propoff_configuration = np.ma.array(arrays)
            self.datarr = acoustic_data_normal_propoff_configuration
            self.explanations = FIELD_EXPLANATIONS

    def fill_in_missing_values(self):
        test_points_array = []

        for test_point_id in  self.datarr['test_point_id']:
            # Add all information from the loaded_data object with the same test_point_id to the acoustic spectrum data
            matching_aerodynamic_data = self.associated_aerodynamic_data.filter(test_point_id=test_point_id)
            datarr_replacement = self.filter(test_point_id=test_point_id).datarr[0].reshape(1, -1)
            # field_to_add = []
            # dtype_to_add = []
            # for field in matching_aerodynamic_data.datarr.dtype.names:
            #     if field in self.datarr.dtype.names:
            #         continue
            #     else:
            #         field_to_add.append(field)
            #datarr_replacement2 = rfn.append_fields(datarr_replacement, field_to_add, [np.full(len(datarr_replacement), matching_aerodynamic_data.datarr[field][0][0], dtype=object) for field in field_to_add], dtypes=object, usemask=True)
            b_extended = rfn.repack_fields(datarr_replacement) #rfn.repack_fields(
                #rfn.append_fields(datarr_replacement, field_to_add, np.zeros(len(field_to_add), dtype='object'))).reshape(-1)
            a_extended = rfn.repack_fields(matching_aerodynamic_data.datarr)  # just to be consistent
            # Now concatenate

            # names_a = set(a_extended.dtype.names)
            # names_b = set(b_extended.dtype.names)
            #
            # all_names = list(names_a | names_b)
            #
            # def align_structured2(arr, all_names):
            #     new_dtype = [(name, arr.dtype[name] if name in arr.dtype.names else float) for name in all_names]
            #     new_arr = np.zeros(arr.shape, dtype=new_dtype)
            #
            #     for name in arr.dtype.names:
            #         new_arr[name] = arr[name]
            #
            #     return new_arr
            #
            # a_aligned = align_structured2(a_extended, all_names)
            # b_aligned = align_structured2(b_extended, all_names)
            #
            # c = np.concatenate([a_aligned, b_aligned])
            old_dtype = b_extended.dtype

            new_dtype = []
            for name in old_dtype.names:
                if name == 'B':
                    new_dtype.append(('B_acoustic', old_dtype[name]))
                elif name == 'dE':
                    new_dtype.append(('dE_acoustic', old_dtype[name]))
                elif name == 'wind_condition':
                    new_dtype.append(('wind_condition_acoustic', old_dtype[name]))
                elif name == 'test_point_id':
                    new_dtype.append(('test_point_id_acoustic', old_dtype[name]))
                else:
                    new_dtype.append((name, old_dtype[name]))


            b_renamed = np.empty(b_extended.shape, dtype=new_dtype)

            for name in old_dtype.names:
                if name == 'B':
                    b_renamed['B_acoustic'] = b_extended[name]
                elif name == 'dE':
                    b_renamed['dE_acoustic'] = b_extended[name]
                elif name == 'wind_condition':
                    b_renamed['wind_condition_acoustic'] = b_extended[name]
                elif name == 'test_point_id':
                    b_renamed['test_point_id_acoustic'] = b_extended[name]
                else:
                    b_renamed[name] = b_extended[name]

            b_extended = b_renamed
            b_extended = rfn.rename_fields(b_extended, {'B': 'B_acoustic', 'dE': 'dE_acoustic', 'wind_condition': 'wind_condition_acoustic', 'test_point_id': 'test_point_id_acoustic'})
            c = rfn.merge_arrays((a_extended, b_extended), flatten=True)
            test_points_array.append(copy.deepcopy(c))
        arrays2 = [row for row in test_points_array]
        newdatarr = np.ma.array(arrays2)
        self.datarr = rfn.drop_fields(newdatarr, ['B_acoustic', 'dE_acoustic', 'wind_condition_acoustic', 'test_point_id_acoustic'])
        self.explanations = FIELD_EXPLANATIONS

        return


    def filter(self, **kwargs):
        return super().filter(acoustic=True, **kwargs)

    def __setitem__(self, key, value):
        return super().__setitem__(key, value, acoustic=True)

    def __array__(self):
        return super().__array__(acoustic=True)

    def __getitem__(self, item):
        return super().__getitem__(item, acoustic=True)

    @property
    def values(self, acoustic=True):
        # Return plain numeric array (for plotting, math, etc.)
        plain = np.ma.column_stack(
            [self.datarr[name] for name in self.datarr.dtype.names]
        )
        out = np.ma.filled(plain, np.nan)
        if not acoustic:
            return out[0][0].ravel()
        else:
            return out.ravel()

class loaded_acoustic_phase_analysis_data(loaded_acoustic_spectrum_data):
    def __init__(self, data_file, configuration, associated_aerodynamic_data):
        self.configuration = configuration
        self.data_file = data_file
        self.associated_aerodynamic_data = associated_aerodynamic_data
        if configuration == 'normal_config':
            self.data_wind_tunnel_test = spio.loadmat(data_file)
            self.phIntp = self.data_wind_tunnel_test['phIntp']
            acoustic_windOn = self.data_wind_tunnel_test['MIC']
            extra_data_windOn = self.data_wind_tunnel_test['opp']
            new_test_points_array = []
            for test_point_acoustic, test_point_extra in zip(acoustic_windOn.squeeze(), extra_data_windOn.squeeze()):
                old_dtype = test_point_acoustic.dtype

                new_dtype = []
                for name in old_dtype.names:
                    # if name == 'B':
                    #     new_dtype.append(('B_acoustic', old_dtype[name]))
                    # elif name == 'dE':
                    #     new_dtype.append(('dE_acoustic', old_dtype[name]))
                    # elif name == 'wind_condition':
                    #     new_dtype.append(('wind_condition_acoustic', old_dtype[name]))
                    # elif name == 'test_point_id':
                    #     new_dtype.append(('test_point_id_acoustic', old_dtype[name]))
                    # else:
                    new_dtype.append((name, old_dtype[name]))

                b_renamed = np.empty(test_point_acoustic.shape, dtype=new_dtype)

                for name in old_dtype.names:
                    # if name == 'B':
                    #     b_renamed['B_acoustic'] = test_point_acoustic[name]
                    # elif name == 'dE':
                    #     b_renamed['dE_acoustic'] = test_point_acoustic[name]
                    # elif name == 'wind_condition':
                    #     b_renamed['wind_condition_acoustic'] = test_point_acoustic[name]
                    # elif name == 'test_point_id':
                    #     b_renamed['test_point_id_acoustic'] = test_point_acoustic[name]
                    # else:
                    b_renamed[name] = test_point_acoustic[name]

                b_extended = b_renamed
                #b_extended = rfn.rename_fields(b_extended, {'B': 'B_acoustic', 'dE': 'dE_acoustic',
                #                                            'wind_condition': 'wind_condition_acoustic',
                #                                            'test_point_id': 'test_point_id_acoustic'})
                test_point_new = rfn.merge_arrays((test_point_extra, b_extended), flatten=True)

                #test_point_new = np.concatenate([a_aligned, b_aligned], axis=1)
                new_test_points_array.append(test_point_new)
            arrays = [[row] for row in new_test_points_array]
            acoustic_phase_data_normal_configuration = np.ma.array(arrays)
            self.datarr = acoustic_phase_data_normal_configuration
            self.explanations = FIELD_EXPLANATIONS
            self.add_test_point_id()
            self.fill_in_missing_values()
        elif configuration == 'tailoff':
            raise NotImplementedError('Phase analysis data has unfortunately not been provided for tailoff configuration')
        else:
            raise NotImplementedError(f"Phase analysis data loading not implemented for configuration '{configuration}'. Are you sure prop is on?")



    def add_test_point_id(self):
        if 'test_point_id' not in self.datarr.dtype.names:
            if self.configuration == 'tailoff':
                self.datarr = add_field(self.datarr, 'test_point_id', 'U25', 'tailoff (from Sinnige)')
                return
            elif self.configuration == 'propoff':
                self.datarr = add_field(self.datarr, 'test_point_id', 'U25', 'propoff (from Sinnige)')
                return
            elif self.configuration == 'modeloff':
                self.datarr = add_field(self.datarr, 'test_point_id', 'U25', 'modeloff (from Sinnige)')
                return
            else:
                self.datarr = add_field(self.datarr, 'test_point_id', 'U25', None)
        self.datarr['test_point_id'][0] = [['32']]
        self.datarr['test_point_id'][1] = [['33']]
        self.datarr['test_point_id'][2] = [['34']]
        self.datarr['test_point_id'][3] = [['35']]
        self.datarr['test_point_id'][4] = [['36']]
        self.datarr['test_point_id'][5] = [['37']]
        self.datarr['test_point_id'][6] = [['38']]
        self.datarr['test_point_id'][7] = [['39']]
        self.datarr['test_point_id'][8] = [['40']]
        self.datarr['test_point_id'][9] = [['Noise1']]
        self.datarr['test_point_id'][10] =[[ 'Noise2']]
        self.datarr['test_point_id'][11] =[[ 'Noise3']]
        self.datarr['test_point_id'][12] =[[ 'Noise4']]

    def fill_in_missing_values(self):
        test_points_array = []

        for test_point_id in  self.datarr['test_point_id']:
            # Add all information from the loaded_data object with the same test_point_id to the acoustic spectrum data
            matching_aerodynamic_data = self.associated_aerodynamic_data.filter(test_point_id=test_point_id)
            datarr_replacement = self.filter(test_point_id=test_point_id).datarr[0].reshape(1, -1)
            # field_to_add = []
            # dtype_to_add = []
            # for field in matching_aerodynamic_data.datarr.dtype.names:
            #     if field in self.datarr.dtype.names:
            #         continue
            #     else:
            #         field_to_add.append(field)
            #datarr_replacement2 = rfn.append_fields(datarr_replacement, field_to_add, [np.full(len(datarr_replacement), matching_aerodynamic_data.datarr[field][0][0], dtype=object) for field in field_to_add], dtypes=object, usemask=True)
            b_extended = rfn.repack_fields(datarr_replacement) #rfn.repack_fields(
                #rfn.append_fields(datarr_replacement, field_to_add, np.zeros(len(field_to_add), dtype='object'))).reshape(-1)
            a_extended = rfn.repack_fields(matching_aerodynamic_data.datarr)  # just to be consistent
            # Now concatenate

            # names_a = set(a_extended.dtype.names)
            # names_b = set(b_extended.dtype.names)
            #
            # all_names = list(names_a | names_b)
            #
            # def align_structured2(arr, all_names):
            #     new_dtype = [(name, arr.dtype[name] if name in arr.dtype.names else float) for name in all_names]
            #     new_arr = np.zeros(arr.shape, dtype=new_dtype)
            #
            #     for name in arr.dtype.names:
            #         new_arr[name] = arr[name]
            #
            #     return new_arr
            #
            # a_aligned = align_structured2(a_extended, all_names)
            # b_aligned = align_structured2(b_extended, all_names)
            #
            # c = np.concatenate([a_aligned, b_aligned])
            old_dtype = b_extended.dtype

            new_dtype = []
            for name in old_dtype.names:
                if name == 'B':
                    new_dtype.append(('B_acoustic', old_dtype[name]))
                elif name == 'dE':
                    new_dtype.append(('dE_acoustic', old_dtype[name]))
                elif name == 'wind_condition':
                    new_dtype.append(('wind_condition_acoustic', old_dtype[name]))
                elif name == 'test_point_id':
                    new_dtype.append(('test_point_id_acoustic', old_dtype[name]))
                elif name == 'AoA':
                    new_dtype.append(('AoA_acoustic', old_dtype[name]))
                elif name == 'AoS':
                    new_dtype.append(('AoS_acoustic', old_dtype[name]))
                elif name == 'J_M1':
                    new_dtype.append(('J_M1_acoustic', old_dtype[name]))
                elif name == 'J_M2':
                    new_dtype.append(('J_M2_acoustic', old_dtype[name]))
                else:
                    new_dtype.append((name, old_dtype[name]))


            b_renamed = np.empty(b_extended.shape, dtype=new_dtype)

            for name in old_dtype.names:
                if name == 'B':
                    b_renamed['B_acoustic'] = b_extended[name]
                elif name == 'dE':
                    b_renamed['dE_acoustic'] = b_extended[name]
                elif name == 'wind_condition':
                    b_renamed['wind_condition_acoustic'] = b_extended[name]
                elif name == 'test_point_id':
                    b_renamed['test_point_id_acoustic'] = b_extended[name]
                elif name == 'AoA':
                    b_renamed['AoA_acoustic'] = b_extended[name]
                elif name == 'AoS':
                    b_renamed['AoS_acoustic'] = b_extended[name]
                elif name == 'J_M1':
                    b_renamed['J_M1_acoustic'] = b_extended[name]
                elif name == 'J_M2':
                    b_renamed['J_M2_acoustic'] = b_extended[name]
                else:
                    b_renamed[name] = b_extended[name]

            b_extended = b_renamed
            b_extended = rfn.rename_fields(b_extended, {'B': 'B_acoustic', 'dE': 'dE_acoustic', 'wind_condition': 'wind_condition_acoustic', 'test_point_id': 'test_point_id_acoustic', 'AoA': 'AoA_acoustic', 'AoS': 'AoS_acoustic', 'J_M1': 'J_M1_acoustic', 'J_M2': 'J_M2_acoustic'})
            c = rfn.merge_arrays((a_extended, b_extended), flatten=True)
            test_points_array.append(copy.deepcopy(c))
        arrays2 = [row for row in test_points_array]
        newdatarr = np.ma.array(arrays2)
        self.datarr = rfn.drop_fields(newdatarr, ['B_acoustic', 'dE_acoustic', 'wind_condition_acoustic', 'test_point_id_acoustic', 'AoA_acoustic', 'AoS_acoustic', 'J_M1_acoustic', 'J_M2_acoustic'])
        self.explanations = FIELD_EXPLANATIONS

        return
    def filter(self, **kwargs):
        returned = super().filter(**kwargs)
        returned.phIntp = self.phIntp
        return returned

    def __getitem__(self, item):
        if item == 'phIntp':
            return self.phIntp
        else:
            returned = super().__getitem__(item)
            returned.phIntp = self.phIntp
            return returned


if __name__ == "__main__":
    # Load the data
    data_normal_configuration = loaded_data('data/normal_config.mat', 'normal_config')
    data_tailoff = loaded_data('data/tailoff.mat', 'tailoff')
    data_propoff = loaded_data('data/propoff.mat', 'propoff')
    data_modeloff = loaded_data('data/modeloff.mat', 'modeloff')

    # print(data_normal_configuration) # print the data
    # print(np.shape(data_normal_configuration)) # perform numpy operations
    # print(np.shape(data_normal_configuration['AoA']))

    #modify data:
    #data_normal_configuration['AoA'] =  1.0  # This will set all AoA values to 1 degree
    # data_normal_configuration['AoA'] = data_normal_configuration['AoA'].values + 1.0  # This will add 1 degree to all AoA values
    
    # Data at elevator deflection 9.5 and 10.5 degrees at wind-on condition, and remove test point 46 (if needed, this is just an example)
    elev_approx_10 = data_normal_configuration.filter(dE__ge=9.5, dE__le=10.5, test_point_id__ne='46', wind_condition='windOn')  # Use 'windOff' for wind-off data
    #print('Available data fields:', [f"{name}: {elev_approx_10.explanations.get(name, 'No description')}" for name in elev_approx_10.datarr.dtype.names])
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
    
    # Example with acoustic data:
    acoustic_spectrum_data_normal_configuration = loaded_acoustic_spectrum_data('data/spectrum_analysis_normal_configuration.mat', 'normal_config', data_normal_configuration)
    acoustic_spectrum_data_propoff = loaded_acoustic_spectrum_data('data/spectrum_analysis_propoff.mat', 'propoff', data_propoff)
    phase_analysis_normal_configuration = loaded_acoustic_phase_analysis_data('data/acoustic_propeller_phase_analysis_normal_condition.mat', 'normal_config', data_normal_configuration)
    acfilt = acoustic_spectrum_data_normal_configuration.filter(AoA__eq=12.)
    frequencies = acfilt['flab'].values
    SPSL = acfilt['SPSL'].values
    plt.figure()
    plt.scatter([freqplot.squeeze() for freqplot in frequencies], [SPSLplot.squeeze() for SPSLplot in SPSL], s=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL (dB)')
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