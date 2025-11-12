import pygambit
import pygambit.nash as gn
import pyomo.environ as pyo

#Solves for NE via support ennumeration
def solve_ne(game: pygambit.gambit.Game):
    return gn.enummixed_solve(game)

#Assumes leader is row, follower is col
#Algorithm taken from Conitzer, V., & Sandholm, T. (2006). Computing the optimal strategy to commit to. ACM Conference on Economics and Computation.
import pyomo.environ as pyo
def solve_stackelberg_equilibrium(leader_u, follower_u):
    # create a model
    l = leader_u.shape[0]
    f = follower_u.shape[1]
    # index for x
    idx = [i for i in range(l)]
    solver = pyo.SolverFactory('glpk')
    leader_strat = None
    follower_strat = None
    best_u = None
    for br in range(f):
        # create decision variables
        model = pyo.ConcreteModel()
        model.x = pyo.Var(idx, domain=pyo.NonNegativeReals)

        # create objective
        model.OBJ = pyo.Objective(expr = sum(model.x[i] * leader_u[i,br] for i in range(l)), sense=pyo.maximize)

        # create constraints
        model.constraints = pyo.ConstraintList()
        model.constraints.add(sum(model.x[i] for i in range(l)) == 1)
        for a in range(f):
            if a != br:
                model.constraints.add(sum(model.x[i] * follower_u[i,a] for i in range(l)) <= sum(model.x[i] * follower_u[i,br] for i in range(l)))

        results = solver.solve(model)
        
        if results.solver.termination_condition == "infeasible":
            continue
        
        lu = model.OBJ()
        if best_u == None or best_u < lu:
            best_u = lu
            leader_strat = [model.x[i]() for i in range(l)]
            follower_strat = [1 if i == br else 0 for i in range(f)]
    return leader_strat, follower_strat
        