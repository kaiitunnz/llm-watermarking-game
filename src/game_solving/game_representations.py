import pygambit 
import numpy as np
from .utils import xUy, seconds_to_hour

class WaterMarkGame:
    # f_w : 1d array of shape w
    # f_a : 2d array of shape w * a
    # u_w : 1d array of shape w
    # u_a : 2d array of shape w * a
    # c_w : 1d array of shape w
    # c_a : 2d array of shape w * a
    def __init__(self, f_w: np.ndarray, f_a: np.ndarray, u_w: np.ndarray, u_a: np.ndarray, c_w: np.ndarray, c_a: np.ndarray):
        self.w = f_a.shape[0]
        self.a = f_a.shape[1]
        self.f_w = f_w
        self.f_a = f_a
        self.u_w = u_w
        self.u_a = u_a
        self.c_w = seconds_to_hour(c_w)
        self.c_a = seconds_to_hour(c_a)

    def to_nf_game(self, a_f, a_uw, a_ua, a_cw, a_ca):
        U_w = self.get_defender_payoffs(a_f, a_uw, a_cw)
        U_a = self.get_attacker_payoffs(a_f, a_ua, a_ca)
        return pygambit.gambit.Game.from_arrays(U_w, U_a)
    
    def _w_f(self, a_f):
        return a_f * (1 - self.get_asr())

    def _a_f(self, a_f):
        return a_f * self.get_asr()
    
    def _w_u(self, a_uw):
        return np.tile(a_uw * self.u_w, (self.a, 1)).transpose()

    def _a_u(self, a_ua):
        return a_ua * self.u_a
    
    def _w_c(self, a_cw):
        return np.tile(a_cw * self.c_w, (self.a, 1)).transpose()

    def _a_c(self, a_ca):
        return a_ca * (self.c_a - np.tile(self.c_w, (self.a, 1)).transpose())

    # payoff to defender is a_f * (1 - f) + a_uw * u_w + a_cw * c_w
    # defender is row player
    def get_defender_payoffs(self, a_f, a_uw, a_cw):
        return self._w_f(a_f) + self._w_u(a_uw) - self._w_c(a_cw)
    
    # payoff to attacker is a_f * f + a_ua * u_a + a_ca * c_a
    # attacker is col player
    def get_attacker_payoffs(self, a_f, a_ua, a_ca):
        return self._a_f(a_f) + self._a_u(a_ua) - self._a_c(a_ca)

    def get_asr(self):
        base_dr = 1 - np.tile(self.f_w, (self.a, 1)).transpose()
        attacked_dr = 1 - self.f_a
        return np.nan_to_num((base_dr - attacked_dr) / base_dr, nan=1)

    def get_payoff(self, defender_strat: np.ndarray, attacker_strat: np.ndarray, a_f, a_uw, a_ua, a_cw, a_ca):
        def_u = xUy(defender_strat, self.get_defender_payoffs(a_f, a_uw, a_cw), attacker_strat)
        att_u = xUy(defender_strat, self.get_attacker_payoffs(a_f, a_ua, a_ca), attacker_strat)
        return def_u, att_u
    
    def get_f_payoff(self, defender_strat: np.ndarray, attacker_strat: np.ndarray, a_f):
        def_u = xUy(defender_strat, self._w_f(a_f), attacker_strat)
        att_u = xUy(defender_strat, self._a_f(a_f), attacker_strat)
        return def_u, att_u
    
    def get_u_payoff(self, defender_strat: np.ndarray, attacker_strat: np.ndarray, a_uw, a_ua):
        def_u = xUy(defender_strat, self._w_u(a_uw), attacker_strat)
        att_u = xUy(defender_strat, self._a_u(a_ua), attacker_strat)
        return def_u, att_u
    
    def get_c_payoff(self, defender_strat: np.ndarray, attacker_strat: np.ndarray, a_cw, a_ca):
        def_u = xUy(defender_strat, self._w_c(a_cw), attacker_strat)
        att_u = xUy(defender_strat, self._a_c(a_ca), attacker_strat)
        return def_u, att_u

