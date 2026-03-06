import numpy as np
from interpolate import interpolate_plots


class BoundaryCorrections:
    
    def __init__(self, CD_0 = 0, V_unc=0, rho=0, q_unc=0, T=0, alpha_unc=0, CL_unc=0, CD_unc=0, CM_c4_unc=0, CL_alpha=0):
        
        self.CD_0 = CD_0
        self.V_unc = V_unc
        self.rho = rho
        self.q_unc = q_unc
        self.T = T
        self.alpha_unc = alpha_unc
        self.CL_unc = CL_unc
        self.CD_unc = CD_unc
        self.CM_c4_unc = CM_c4_unc
        self.CL_alpha = CL_alpha
        
        self.t_over_c = 15.824 / 100
        
        self.V_model = self._calc_model_volume()
        self.C_tunnel = self._calc_tunnel_cross_section()
        self.S_model = self._calc_model_reference_area()
        self.S_prop = self._calc_propeller_area()
        self.c_mac, self.l_tail = self._calc_tail_distance()
        self.K1, self.K3, self.tau1, self.tau2, self.delta = self._get_interpolated_constants()
    
    def _get_interpolated_constants(self):
        inty = interpolate_plots()
        K1 = inty.get_K1(66, self.t_over_c) # Look for which airfoil series are required
        print(K1)
        K3 = inty.get_K3(self.t_over_c)
        print(K3)
        tau1 = 1
        tau2 = 1
        delta = 1
        return K1, K3, tau1, tau2, delta
    
    @staticmethod
    def _calc_model_volume():
        V_fuselage = 0.0160632
        V_aft_strut = 0.0004491
        V_wing_struts = 0.0035296 # Both struts together
        V_wing = 0.0030229
        V_HT = 0.0009751
        V_nacelle = 2 * 0.0007921 # per nacelle
        V_VT = 0.0003546
        return V_fuselage + V_aft_strut + V_wing_struts + V_wing + V_HT + V_nacelle + V_VT
    
    @staticmethod
    def _calc_tunnel_cross_section():
        h_tunnel = 1.25
        w_tunnel = 1.80
        A_tunnel = h_tunnel * w_tunnel
        triangle_side = 0.3
        A_triangle = 0.5 * triangle_side * triangle_side
        A_total_triangles = 4 * A_triangle
        return A_tunnel - A_total_triangles
    
    @staticmethod
    def _calc_model_reference_area():
        d_fuselage = 0.140
        l_fuselage = 1.342
        A_fuselage = d_fuselage * l_fuselage
        A_wing = 0.2172
        A_HT = 0.0858
        A_VT = 0.0415
        return A_fuselage + A_wing + A_HT + A_VT
    
    @staticmethod
    def _calc_propeller_area():
        D_prop = 0.2032
        return (np.pi / 4) * D_prop**2
    
    @staticmethod
    def _calc_tail_distance():
        c_mac = 0.165
        ratio = 0.5313
        return c_mac, ratio / c_mac
    
    def apply_boundary_corrections(self):
        
        # Solid Blockage
        epsilon_sb_f = (self.K3 * self.tau1 * self.V_model) / (self.C_tunnel**(3/2))
        epsilon_sb_w = (self.K1 * self.tau1 * self.V_model) / (self.C_tunnel**(3/2))
        epsilon_sb = epsilon_sb_f + epsilon_sb_w
        
        # Wake Blockage
        epsilon_wb_0 = (self.S_model / (4 * self.C_tunnel)) * self.CD_0
        epsilon_wb_s = 0 # Attached flow assumption -> epsilon_wb_s = ((5 * self.S_model) / (4 * self.C_tunnel)) * (CD_unc - CD_0 - CD_i)
        epsilon_wb = epsilon_wb_0 + epsilon_wb_s
        
        # Slipstream Blockage
        T_C_star = self.T / (self.rho * self.V_unc**2 * self.S_prop)
        epsilon_ss = -((T_C_star) / (2 * np.sqrt(1 + (2 * T_C_star)))) * (self.S_prop / self.C_tunnel)
        
        # Lift interference
        delta_alpha = self.delta * (self.S_model / self.C_tunnel) * self.CL_unc * (1 + self.tau2)
        delta_alpha_sc = ((self.tau2 * (0.5 * self.c_mac)) / (1 + self.tau2 * (0.5 * self.c_mac))) * delta_alpha
        delta_CD_wing = self.delta * (self.S_model / self.C_tunnel) * self.CL_unc**2
        delta_CM_c4_wing = (1/8) * delta_alpha_sc * CL_alpha
        
        # Downwash correction
        delta_alpha_tail = self.delta * (self.S_model / self.C_tunnel) * self.CL_unc * (1 + self.tau2 * self.l_tail)
        delta_CM_c4_tail = delta_CM_c4_wing * delta_alpha_tail
        
        # Total corrections
        epsilon = epsilon_sb + epsilon_wb + epsilon_ss
        alpha_cor = self.alpha_unc + delta_alpha
        V_cor = self.V_unc * (1 + epsilon)
        q_cor = self.q_unc * (1 + epsilon)**2
        CL_cor = self.CL_unc * (1 + epsilon)**(-2)
        CD_cor = self.CD_unc * (1 + epsilon)**(-2) + delta_CD_wing
        CM_c4_cor = self.CM_c4_unc * (1 + epsilon)**(-2) + delta_CD_wing + delta_CM_c4_tail
        
        return alpha_cor, V_cor, q_cor, CL_cor, CD_cor, CM_c4_cor


if __name__ == "__main__":
    bc = BoundaryCorrections()