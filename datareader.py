import scipy.io as spio
import numpy as np

FIELD_EXPLANATIONS = {
            'AoA': 'Angle of Attack in degrees',
            'AoS': 'Angle of Sideslip in degrees',
            'CL': 'Lift Coefficient',
            'CD': 'Drag Coefficient',
            'CYaw': 'Side Force Coefficient',
            'CMroll': 'Rolling Moment Coefficient',
            'CMpitch': 'Pitching Moment Coefficient',
            'CMpitch25c': 'Pitching Moment Coefficient at 25% chord',
            'CMyaw': 'Yawing Moment Coefficient',
            'rho': 'Air Density in kg/m^3. Corrected from measured value by the Matlab code to account for sensor calibration.',
            'V': 'Freestream Velocity in m/s. Corrected from measured value by the Matlab code to account for sensor calibration.',
            'pInf': 'Freestream Pressure in Pa. Corrected from measured value by the Matlab code to account for sensor calibration.',
            'q': 'Dynamic Pressure in Pa. Corrected from measured value by the Matlab code to account for sensor calibration.',
            'temp': 'Air Temperature in K',
            'nu': 'Kinematic Viscosity in m^2/s. Calculated from corrected parameters.',
            'Re': 'Reynolds Number. Corrected from measured value by the Matlab code to account for sensor calibration.',
            'J_M1': 'Advance Ratio of Motor 1',
            'J_M2': 'Advance Ratio of Motor 2',
            'rpsM1': 'Rotations per second of Motor 1',
            'rpsM2': 'Rotations per second of Motor 2',
            'iM1': 'Current of Motor 1 in Amperes',
            'iM2': 'Current of Motor 2 in Amperes',
            'tM1': 'Temperature of Motor 1 in Celsius',
            'tM2': 'Temperature of Motor 2 in Celsius',
            'vM1': 'Voltage of Motor 1 in Volts',
            'vM2': 'Voltage of Motor 2 in Volts',
            'b': 'Wing Span in meters',
            'S': 'Wing Planform Area in square meters',
            'c': 'Mean Aerodynamic Chord in meters',
            'elevator_deflection': 'Elevator Deflection in degrees',
            'wind_condition': 'Wind condition during the test, either "windOn" or "windOff"'
        }

def add_field(a, name, dtype, value):
    b = np.zeros(a.shape, a.dtype.descr + [(name, dtype)])
    b[list(a.dtype.names)] = a
    if np.isscalar(value) or value == None:
        b[name] = value
    else:
        b[name][...] = value
    return b
def align_dtype(arr, target_dtype):
    out = np.ma.empty(arr.shape, dtype=target_dtype)
    out.mask = True
    for name in arr.dtype.names:
        out[name] = arr[name]
    out.mask = True
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
            prototype_dtype = add_field(add_field(windOn[0][0]['G31_d0'], 'elevator_deflection',
                                                                       np.float64, 0), 'wind_condition', 'U10', 'windOn').dtype


            data_normal_configuration = np.concatenate([add_field(add_field(windOn[0][0]['G31_d0'], 'elevator_deflection',
                                                                       np.float64, 0), 'wind_condition', 'U10', 'windOn'),
                                         align_dtype(add_field(add_field(windOff[0][0]['G31_d0'], 'elevator_deflection'
                                                                        , np.float64, 0), 'wind_condition', 'U10', 'windOff'), prototype_dtype),
                                         add_field(add_field(windOn[0][0]['G31_de_n10'], 'elevator_deflection',
                                                                 np.float64, -10), 'wind_condition', 'U10', 'windOn'),
                                         align_dtype(add_field(add_field(windOff[0][0]['G31_de_n10'], 'elevator_deflection',
                                                                  np.float64, -10), 'wind_condition', 'U10', 'windOff'), prototype_dtype),
                                         add_field(add_field(windOn[0][0]['G31_de_20'], 'elevator_deflection',
                                                                 np.float64, 20), 'wind_condition', 'U10', 'windOn'),
                                         align_dtype(add_field(add_field(windOff[0][0]['G31_de_20'], 'elevator_deflection',
                                                                  np.float64, 20), 'wind_condition', 'U10', 'windOff'), prototype_dtype),
                                         add_field(add_field(windOn[0][0]['G31_den10'], 'elevator_deflection',
                                                                 np.float64, 10), 'wind_condition', 'U10', 'windOn'),
                                         align_dtype(add_field(add_field(windOff[0][0]['G31_den10'], 'elevator_deflection',
                                                                  np.float64, 10), 'wind_condition', 'U10', 'windOff'), prototype_dtype),
                                         add_field(add_field(windOn[0][0]['G31_de_n20'], 'elevator_deflection',
                                                                    np.float64, -20), 'wind_condition', 'U10', 'windOn'),
                                            align_dtype(add_field(add_field(windOff[0][0]['G31_de_n20'], 'elevator_deflection',
                                                                    np.float64, -20), 'wind_condition', 'U10', 'windOff'), prototype_dtype)])
            data_normal_configuration.explanations = FIELD_EXPLANATIONS
            self.datarr = data_normal_configuration

        elif configuration == 'tailoff':
            data_wind_tunnel_test = spio.loadmat(data_file)
            tailOff_windOn = data_wind_tunnel_test['BAL']['windOn'][0][0]['tailOff_beta0_balance'][0][0]
            tailOff_windOff = data_wind_tunnel_test['BAL']['windOff'][0][0]['tailOff_beta0_balance'][0][0]
            prototype_dtype = add_field(add_field(tailOff_windOn, 'elevator_deflection',
                                                                       np.float64, None), 'wind_condition', 'U10', 'windOn').dtype
            data_tailoff = np.concatenate([add_field(add_field(tailOff_windOn, 'elevator_deflection',
                                                                       np.float64, None), 'wind_condition', 'U10', 'windOn'),
                                         align_dtype(add_field(add_field(tailOff_windOff, 'elevator_deflection'
                                                                        , np.float64, None), 'wind_condition', 'U10', 'windOff'), prototype_dtype)])
            data_tailoff.explanations = FIELD_EXPLANATIONS
            self.datarr = data_tailoff





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
            mask = mask & condition

        # Return filtered masked array
        filtarr = self.datarr[mask]
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
        return self.datarr

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
        retarr = self.datarr
        plain = np.ma.column_stack(
            [self.datarr[name] for name in self.datarr.dtype.names]
        )
        out = np.ma.filled(plain, np.nan)
        return out[0][0].ravel()






if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_normal_configuration = loaded_data('normal_config.mat', 'normal_config')
    print(data_normal_configuration)
    data_tailoff = loaded_data('tailoff.mat', 'tailoff')
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