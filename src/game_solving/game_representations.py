import pygambit 
import numpy as np

class WaterMarkGame:
    # f_w : 1d array of shape w
    # f_a : 2d array of shape w * a
    # u_w : 1d array of shape w
    # u_a : 1d array of shape a
    # c_w : 1d array of shape w
    # c_a : 1d array of shape a
    def __init__(self, f_w: np.ndarray, f_a: np.ndarray, u_w: np.ndarray, u_a: np.ndarray, c_w: np.ndarray, c_a: np.ndarray):
        self.w = f_a.shape[0]
        self.a = f_a.shape[1]
        self.f_w = f_w
        self.f_a = f_a
        self.u_w = u_w
        self.u_a = u_a
        self.c_w = c_w
        self.c_a = c_a

    def to_nf_game(self, a_f, a_uw, a_ua, a_cw, a_ca):
        U_w = self.get_defender_payoffs(a_f, a_uw, a_cw)
        U_a = self.get_attacker_payoffs(a_f, a_ua, a_ca)
        return pygambit.gambit.Game.from_arrays(U_w, U_a)
    
    # payoff to defender is a_f * (1 - f) + a_uw * u_w + a_cw * c_w
    # defender is row player
    def get_defender_payoffs(self, a_f, a_uw, a_cw):
        return a_f * (1 - self.get_asr()) + np.tile(a_uw * self.u_w, (self.a, 1)).transpose() + np.tile(a_cw * self.c_w, (self.a, 1)).transpose()
    
    # payoff to attacker is a_f * f + a_ua * u_a + a_ca * c_a
    # attacker is col player
    def get_attacker_payoffs(self, a_f, a_ua, a_ca):
        return a_f * self.get_asr() + np.tile(a_ua * self.u_a, (self.w, 1)) + np.tile(a_ca * self.c_a, (self.w, 1))

    def get_asr(self):
        base_dr = 1 - np.tile(self.f_w, (self.a, 1)).transpose()
        attacked_dr = 1 - self.f_a
        return np.nan_to_num((base_dr - attacked_dr) / base_dr, nan=1)

    def get_payoff(self, defender_strat: np.ndarray, attacker_strat: np.ndarray, a_f, a_uw, a_ua, a_cw, a_ca):
        def_u = xUy(defender_strat, self.get_defender_payoffs(self, a_f, a_uw, a_cw), attacker_strat)
        att_u = xUy(defender_strat, self.get_attacker_payoffs(self, a_f, a_ua, a_ca), attacker_strat)
        return def_u, att_u
    
    def get_f_payoff(self, defender_strat: np.ndarray, attacker_strat: np.ndarray, a_f):
        def_u = xUy(defender_strat, a_f * (1 - self.get_asr()), attacker_strat)
        att_u = xUy(defender_strat, a_f * self.get_asr(), attacker_strat)
        return def_u, att_u
    
    def get_u_payoff(self, defender_strat: np.ndarray, attacker_strat: np.ndarray, a_uw, a_ua):
        def_u = xUy(defender_strat, np.tile(a_uw * self.u_w, (self.a, 1)).transpose(), attacker_strat)
        att_u = xUy(defender_strat, np.tile(a_ua * self.u_a, (self.w, 1)), attacker_strat)
        return def_u, att_u
    
    def get_c_payoff(self, defender_strat: np.ndarray, attacker_strat: np.ndarray, a_cw, a_ca):
        def_u = xUy(defender_strat, np.tile(a_cw * self.c_w, (self.a, 1)).transpose(), attacker_strat)
        att_u = xUy(defender_strat, np.tile(a_ca * self.c_a, (self.w, 1)), attacker_strat)
        return def_u, att_u

def xUy(x, U, y):
    return np.dot(np.dot(x, U), y)