import numpy as np
from scipy.integrate import solve_ivp

def simulate(x0, f, dt, t_end, u_fun=None, **kwargs):
    """Simulate a system of ODEs: dx/dt = f(x).
        Parameters:
        - x0 (ndarray): Initial condition.
        - f (function): Function that defines the ODEs.
        - dt (float): Time step.
        - t_end (float): Final time. 
        Returns:
        - ndarray: Simulation results.
    """    
    error = 1 - (t_end % dt) / dt 
    if error > 1e-6 and (1-error > 1e-6):
        raise ValueError("t_end must be a multiple of dt.")
        
    t_span = (0, t_end)
    t_eval = np.arange(0, t_end+dt, dt)
    if u_fun is None:
        fun = lambda t, x: f(t, x)
    else:
        fun = lambda t, x: f(t, x, u_fun=u_fun)
    
    sol = solve_ivp(fun=fun, t_span=t_span, y0=x0, t_eval=t_eval, **kwargs)
    
    if u_fun is None:
        return sol.y.T, sol.t.T
    else:
        u_values = np.array([u_fun(t) for t in sol.t.T])
        return sol.y.T, u_values.reshape(-1, len(u_fun(0))), sol.t.T
