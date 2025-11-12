import numpy as np

def xUy(x, U, y):
    return np.dot(np.dot(x, U), y)

SECONDS_IN_HOUR = 3600.0

def seconds_to_hour(t):
    return t / SECONDS_IN_HOUR

def get_strat_from_ne(ne):
    strategy_profile = []
    for p in ne.game.players:
        strategy = ne[p]
        strat = np.array([float(strategy[s]) for s in p.strategies])
        strategy_profile.append(strat)
    return strategy_profile