import scipy.io as spio
import numpy as np
import copy

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
            'elevator_deflection': 'Elevator Deflection in degrees. Not applicable for tail off. Not calculated for propoff',
            'wind_condition': 'Wind condition during the test, either "windOn" or "windOff". Unknown for propoff',
            'dE': 'We need to ask Thomas Sinnige what this is',
            'dR': 'We need to ask Thomas Sinnige what this is',
        }

def add_field(a, name, dtype, value):
    b = np.zeros(a.shape, a.dtype.descr + [(name, object)])
    b[list(a.dtype.names)] = a
    if np.isscalar(value) or value == None:
        b[name][...] = [[np.full_like(b['AoA'][0][0], np.array([[value]]), dtype=dtype)]]
    else:
        raise NotImplementedError
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
        if configuration == 'normal_config':
            data_wind_tunnel_test = spio.loadmat(data_file)
            windOn = data_wind_tunnel_test['BAL']['windOn'][0][0]
            windOff = data_wind_tunnel_test['BAL']['windOff'][0][0]
            config = data_wind_tunnel_test['BAL']['config'][0][0]
            data_normal_configuration2 = copy.deepcopy(add_field(add_field(windOn[0][0]['G31_d0'], 'elevator_deflection',
                                                              np.float64, 0), 'wind_condition', 'U10', 'windOn'))
            prototype_dtype = data_normal_configuration2.dtype
            data_normal_configuration = copy.deepcopy(data_normal_configuration2)

            data_normal_configuration_array = [align_dtype(add_field(add_field(windOn[0][0]['G31_d0'], 'elevator_deflection',
                                                                       np.float64, 0), 'wind_condition', 'U10', 'windOn'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOff[0][0]['G31_d0'], 'elevator_deflection'
                                                                        , np.float64, 0), 'wind_condition', 'U10', 'windOff'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOn[0][0]['G31_de_n10'], 'elevator_deflection',
                                                                 np.float64, -10), 'wind_condition', 'U10', 'windOn'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOff[0][0]['G31_de_n10'], 'elevator_deflection',
                                                                  np.float64, -10), 'wind_condition', 'U10', 'windOff'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOn[0][0]['G31_de_20'], 'elevator_deflection',
                                                                 np.float64, 20), 'wind_condition', 'U10', 'windOn'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOff[0][0]['G31_de_20'], 'elevator_deflection',
                                                                  np.float64, 20), 'wind_condition', 'U10', 'windOff'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOn[0][0]['G31_den10'], 'elevator_deflection',
                                                                 np.float64, 10), 'wind_condition', 'U10', 'windOn'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOff[0][0]['G31_den10'], 'elevator_deflection',
                                                                  np.float64, 10), 'wind_condition', 'U10', 'windOff'), prototype_dtype, data_normal_configuration2),
                                         align_dtype(add_field(add_field(windOn[0][0]['G31_de_n20'], 'elevator_deflection',
                                                                    np.float64, -20), 'wind_condition', 'U10', 'windOn'), prototype_dtype, data_normal_configuration2),
                                            align_dtype(add_field(add_field(windOff[0][0]['G31_de_n20'], 'elevator_deflection',
                                                                    np.float64, -20), 'wind_condition', 'U10', 'windOff'), prototype_dtype, data_normal_configuration2)]



            for field in prototype_dtype.names:
                arrs_for_concatenation = []
                for indd, arrr in enumerate(data_normal_configuration_array):
                    if type(arrr[field][0][0]) == np.ma.MaskedArray:
                        arrs_for_concatenation.append(arrr[field][0][0])
                    else:
                        arrs_for_concatenation.append(np.ma.masked_array(arrr[field][0][0], mask=np.zeros_like(arrr[field][0][0], dtype=bool)))
                data_normal_configuration[field][0][0] = np.ma.concatenate(arrs_for_concatenation)
            data_normal_configuration = np.ma.array(data_normal_configuration, mask=np.zeros_like(data_normal_configuration, dtype=bool))
            data_normal_configuration.explanations = FIELD_EXPLANATIONS
            self.datarr = data_normal_configuration

        elif configuration == 'tailoff':
            data_wind_tunnel_test = spio.loadmat(data_file)
            tailOff_windOn = data_wind_tunnel_test['BAL']['windOn'][0][0]['tailOff_beta0_balance'][0][0]
            tailOff_windOff = data_wind_tunnel_test['BAL']['windOff'][0][0]['tailOff_beta0_balance'][0][0]
            data_tailoff2 = copy.deepcopy(add_field(add_field(tailOff_windOn, 'elevator_deflection',
                                                                       np.float64, None), 'wind_condition', 'U10', 'windOn'))
            prototype_dtype = data_tailoff2.dtype
            data_tailoff = copy.deepcopy(data_tailoff2)

            data_tailoff_array = [align_dtype(data_tailoff2, prototype_dtype, data_tailoff2), align_dtype(add_field(add_field(tailOff_windOff, 'elevator_deflection'
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
            data_tailoff.explanations = FIELD_EXPLANATIONS
            self.datarr = data_tailoff
        elif configuration == 'propoff':
            ## TODO: PROPOFF NEEDS TO BE MATLAB-Corrected still!! Ask prof if it needs to be corrected because it is already in coefficient form. Also 4409 data points is bit large
            data_wind_tunnel_test = spio.loadmat(data_file)
            propOff_windOn = data_wind_tunnel_test['propOff']
            data_propoff = np.ma.masked_array(add_field(add_field(propOff_windOn, 'elevator_deflection',
                                                                       np.float64, None), 'wind_condition', 'U10', 'windOn'))
            data_propoff.explanations = FIELD_EXPLANATIONS
            self.datarr = data_propoff

        elif configuration == 'modeloff':
            data_wind_tunnel_test = spio.loadmat(data_file)
            modelOff_windOn = data_wind_tunnel_test['modelOff']
            data_modeloff = np.ma.masked_array(add_field(add_field(modelOff_windOn, 'elevator_deflection',
                                                                       np.float64, None), 'wind_condition', 'U10', 'windOn'))
            data_modeloff.explanations = FIELD_EXPLANATIONS
            self.datarr = data_modeloff





    def __getitem__(self, item):
        #retarr = [test_point[item] for test_point in self.datarr]
        retobj = object.__new__(loaded_data)
        retobj.datarr = self.datarr[[item]]
        retobj.datarr.explanations = FIELD_EXPLANATIONS
        retobj.data_file = self.data_file
        return  retobj #np.ma.array(retarr).ravel()

    def filter(self, **kwargs):
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
        filtered = data.filter(wind_condition='windOn', elevator_deflection=20)

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

            field_data = self.datarr[field_name][0][0]

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
        filtarr = copy.deepcopy(self.datarr)
        for field in self.datarr.dtype.names:
            if np.shape(self.datarr[field][0][0])[1] != 1:
                maskt = np.repeat(mask, np.shape(self.datarr[field][0][0])[1], axis=1)
                filtarr[field] = [[self.datarr[field][0][0][maskt]]]
            else:
                filtarr[field] = [[self.datarr[field][0][0][mask]]]
        filtarr.explanations = FIELD_EXPLANATIONS
        retobj = object.__new__(loaded_data)
        retobj.datarr = filtarr
        retobj.data_file = self.data_file
        return retobj

    def __setitem__(self, key, value):
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
            self.datarr[key] = value
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
                raise ValueError(f"Array shape {value_arr.shape} not compatible. "
                               f"Expected (1, {len(self.datarr)}) or ({len(self.datarr)},)")
        elif value_arr.ndim != 1:
            raise ValueError(f"Array must be 1D or 2D, got {value_arr.ndim}D")

        # Check size matches
        if len(value_arr) != len(self.datarr):
            raise ValueError(f"Array size {len(value_arr)} doesn't match data size {len(self.datarr)}. "
                           f"Expected size ({len(self.datarr)},) or (1, {len(self.datarr)})")

        # Set the field
        self.datarr[key] = value_arr

    def __array__(self):
        returner = []
        for field in self.datarr.dtype.names:
            if np.shape(self.datarr[field][0][0])[1] != 1:
                for ind in range(np.shape(self.datarr[field][0][0])[1]):
                    returner.append(self.datarr[field][0][0][:, ind].reshape(-1, 1))
            else:
                returner.append(self.datarr[field][0][0])
        returner = np.array(returner)
        return returner

    def __str__(self):
        fields = self.datarr.dtype.names
        rows_str = []

        for i, row in enumerate(self.datarr):
            rows_str.append(f"Record {i}:")
            for f in fields:
                val = row[f]
                try:
                    explen = self.datarr.explanations[f]
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
    def values(self):
        # Return plain numeric array (for plotting, math, etc.)
        plain = np.ma.column_stack(
            [self.datarr[name] for name in self.datarr.dtype.names]
        )
        out = np.ma.filled(plain, np.nan)
        return out[0][0].ravel()






if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_normal_configuration = loaded_data('normal_config.mat', 'normal_config')
    np.shape(data_normal_configuration)
    #print(data_normal_configuration)
    data_tailoff = loaded_data('tailoff.mat', 'tailoff')
    data_propoff = loaded_data('propoff.mat', 'propoff')
    data_modeloff = loaded_data('modeloff.mat', 'modeloff')
    print(data_tailoff)
    # Example, data at elevator deflection of 20 degrees
    elev_20 = data_normal_configuration.filter(elevator_deflection=20, wind_condition='windOn')  # Use 'windOff' for wind-off data
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
    print('Available data fields:', [f'{name}: {elev_20.datarr.explanations.get(name, 'No description')}' for name in elev_20.datarr.dtype.names])
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