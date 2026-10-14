import numpy as np

N_VAR = 5
BOUNDS = [(-5.0, 5.0)] * N_VAR
XL = np.full(N_VAR, -5.0)
XU = np.full(N_VAR, 5.0)
WEIGHTS = np.linspace(0.0, 1.0, 201)[1:-1]  # 199 interior weights
SLSQP_OPTIONS = {'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
SEED = 42

# Common mathematical problem: g_leq(x) <= 0, h1(x)=h2(x)=0.
def f1(x):
    x = np.asarray(x)
    return np.sum(x**2, axis=-1)

def f2(x):
    x = np.asarray(x)
    return 3*x[...,0] + 2*x[...,1] - x[...,2]/3.0 + 0.01*(x[...,3]-x[...,4])**3

def g_leq(x):
    x = np.asarray(x)
    return np.sum(x[..., :4]**2, axis=-1) - 10.0

def h1(x):
    x = np.asarray(x)
    return 4*x[...,0] - 2*x[...,1] + 0.8*x[...,2] + 0.6*x[...,3] + 0.5*x[...,4]**2

def h2(x):
    x = np.asarray(x)
    return x[...,0] + 2*x[...,1] - x[...,2] - 0.5*x[...,3] + x[...,4] - 2.0

def scipy_constraints(extra=None):
    cons = [
        {'type': 'ineq', 'fun': lambda x: -g_leq(x)},  # SciPy wants >= 0
        {'type': 'eq', 'fun': h1},
        {'type': 'eq', 'fun': h2},
    ]
    if extra:
        cons.extend(extra)
    return cons

def diagnostics(x):
    x = np.asarray(x, dtype=float)
    return {
        'g': float(g_leq(x)),
        'h1': float(h1(x)),
        'h2': float(h2(x)),
        'max_abs_h': float(max(abs(h1(x)), abs(h2(x)))),
        'max_abs_x': float(np.max(np.abs(x))),
        'bound_margin': float(5.0 - np.max(np.abs(x))),
    }
