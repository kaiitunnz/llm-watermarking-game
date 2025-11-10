import pygambit 
import numpy as np

class NormalFormWaterMarkGame:
    # f : 2d array of shape w * a
    # u_w : 1d array of shape w
    # u_a : 1d array of shape a
    # c_w : 1d array of shape w
    # c_a : 1d array of shape a
    def __init__(self, f: np.ndarray, u_w: np.ndarray, u_a: np.ndarray, c_w: np.ndarray, c_a: np.ndarray):
        self.w = f.shape[0]
        self.a = f.shape[1]
        self.f = f
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
        return a_f * np.copy(1 - self.f) + np.tile(a_uw * self.u_w, (self.a, 1)).transpose() + np.tile(a_cw * self.c_w, (self.a, 1)).transpose()
    
    # payoff to attacker is a_f * f + a_ua * u_a + a_ca * c_a
    # attacker is col player
    def get_attacker_payoffs(self, a_f, a_ua, a_ca):
        return a_f * np.copy(self.f) + np.tile(a_ua * self.u_a, (self.w, 1)) + np.tile(a_ca * self.c_a, (self.w, 1))


