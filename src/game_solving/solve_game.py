import pygambit
import pygambit.nash as gn

#Solves for NE via support ennumeration
def solve_normal_form_ne(game: pygambit.gambit.Game):
    return gn.enummixed_solve(game)