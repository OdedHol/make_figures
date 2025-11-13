import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import numpy as np

class SGH_class:
    def __init__(self, a=0.05, c=0.2, k=1, d=0.01, p=1.5,
                       e0=5, n=0.4, zr=300,
                       ksat=800, gamma=10, beta_a=1/3, beta_e=4,
                       lambda_h=-2, lambda_fc=10, S_h=0.3, S_fc=0.6,
                       infilt_alpha=0.1, infilt_i0=0.2,
                       tree_root_active=0, shading_active=1, logistic_active=1, infiltration_active=0,
                       dark_et=0.0, initial_guess=(0.9, 0.5), fsolve_iterations=20,
                       t_eval_points=300, t_span_max=50000,
                       **kwargs):
        
        # State: Initial guess (y) is now purely a parameter stored in the dict
        self.params = {
            # Core Parameters
            'a':a,      # Assimilation (1/d)
            'c':c,      # Tree Cover (kg/m^2)
            'k':k,      # Max Biomass (kg/m^2)
            'd':d,      # Death (1/d)
            'p':p,      # Precipitation Rate (mm/day)
            'e0':e0,    # Evaportaion Rate (mm*kg/d*m^2 )
            'n':n,      # Porosity (-)
            'zr':zr,    # Root depth (mm) 
            'ksat':ksat, # Saturated Hydraulic Conductivity (mm/day)
            'dark_et': dark_et, # Dark evapotranspiration
            'infilt_alpha': infilt_alpha, # Infiltration parameter alpha
            'infilt_i0': infilt_i0,       # Infiltration parameter i0
            
            # Non-Linearity/Constant Parameters
            'gamma':gamma,      # Exponent of leakage
            'beta_a':beta_a,    # Assimilation exponent
            'beta_e':beta_e,    # Transpiration exponent
            'lambda_h':lambda_h, # Tree water constant (Hygroscopic)
            'lambda_fc':lambda_fc, # Tree water constant (Field Capacity)
            'S_h': S_h,          # Hygroscopic point (Soil Saturation)
            'S_fc': S_fc,        # Field capacity (Soil Saturation)
            
            # Boolean/Switch Flags
            'tree_root_active': tree_root_active, 
            'shading_active': shading_active, 
            'logistic_active': logistic_active,
            'infiltration_active': infiltration_active,
            
            # Solver Parameters
            'initial_guess': initial_guess, 
            'fsolve_iterations':fsolve_iterations,
            't_span_max': t_span_max,
            't_eval_points': t_eval_points,
        }
        self.params.update(kwargs)

    # --- Parameter Management ---

    def backup_params(self):
        """Backs up the current model parameters."""
        self._backup_params = self.params.copy()

    def restore_params(self):
        """Restores model parameters to the backed up state."""
        if hasattr(self, "_backup_params"):
            self.params = self._backup_params.copy()
        else:
            raise AttributeError("No backup found. Please backup parameters before restoration.")

    # --- Internal Utility ---

    def _get_random_guess(self):
        """Generates a random initial guess (B, S) for fsolve."""
        # Note: B and S range constants should ideally be in self.params
        B_rand = np.random.uniform(0.5, 1.5)
        S_rand = np.random.uniform(0.5, 0.6)
        return (B_rand, S_rand)
    
    # --- Component Functions (Public for Plotting/External Use) ---
    
    # All component functions accept (B, S) and optional **kwargs 
    # to allow plotting with different parameter values without affecting self.params
    
    def drainage(self, B, S, **kwargs):
        p = {**self.params, **kwargs} # Merge params for temporary overrides
        return p['ksat'] * S ** p['gamma']

    def transpiration(self, B, S, **kwargs):
        p = {**self.params, **kwargs}
        return p['e0'] * B * S
  
    def shading_assimilation(self, B, S, **kwargs):
        p = {**self.params, **kwargs}
        fa = (1 - p['shading_active'] * p['c']) ** p['beta_a']
        return fa

    def shading_transpiration(self, B, S, **kwargs):
        p = {**self.params, **kwargs}
        fe = (1 - p['shading_active'] * p['c'])**p['beta_e']
        fe = (1 - p['dark_et']) * fe + p['dark_et'] 
        if p['infiltration_active']:
            fe = 1
        return fe
    
    def infiltration_factor(self, B, S, **kwargs):
        p = {**self.params, **kwargs}
        a = p['infilt_alpha']
        i0 = p['infilt_i0']
        if p['infiltration_active']:
            fi = (p['c'] + a*i0) / (p['c'] + a)
        else:
            fi = 1
        return fi

    def relu(self, x):
        """ReLU activation function: max(0, x)."""
        return np.maximum(0, x)

    def tree_root_function(self, B, S, **kwargs):
        p = {**self.params, **kwargs}
        m = (p['lambda_fc'] - p['lambda_h']) / (p['S_fc'] - p['S_h'])
        
        # Calculate tree root water uptake factor (fr)
        fr = p['c'] * (p['lambda_h'] + m * self.relu(S - p['S_h']) - m * self.relu(S - p['S_fc']))
        
        return fr * p['tree_root_active']

    def carrying_capacity(self, B, S, **kwargs):
        p = {**self.params, **kwargs}
        return (1 - p['logistic_active']*B / p['k'])
           
    def growth_biomass(self, B, S, **kwargs):
        p = {**self.params, **kwargs}
        # Note: B and S are not directly used here but kept for consistent signature
        return p['a'] * self.shading_assimilation(B, S, **kwargs) * S * B * self.carrying_capacity(B, S, **kwargs)
        
    def death_biomass(self, B, S, **kwargs):
        p = {**self.params, **kwargs}
        return B * p['d']

    def evapotranspiration(self, B, S, **kwargs):
        p = {**self.params, **kwargs}
        return p['e0'] * self.shading_transpiration(B, S, **kwargs) * B * S
    
    def analytical_solutions_no_carrying_capacity(self, **kwargs):
        p = {**self.params, **kwargs}
        # Analytical steady-state solutions without carrying capacity
        S_star = p['d'] / (self.shading_assimilation(None, None, **p) * p['a'])
        B_star = (p['p']*self.infiltration_factor(None, None, **kwargs) - self.tree_root_function(None, S_star, **p) - self.drainage(None, S_star, **p)) / (p['e0'] * self.shading_transpiration(None, None, **p) * S_star)
        return np.maximum(B_star, 0), S_star
    
    # --- Main Equation Solvers ---
    
    def equ(self, y, **kwargs):
        """
        The core system of ODEs, giving [dB/dt, dS/dt].
        
        :param y: The state vector (B, S).
        :param kwargs: Optional parameter overrides.
        :return: np.array([dB/dt, dS/dt])
        """
        B, S = y
        p = {**self.params, **kwargs} # Merge params for temporary overrides
        
        # Pass B, S, and kwargs through to component functions
        dbdt = self.growth_biomass(B, S, **kwargs) - self.death_biomass(B, S, **kwargs)

        dsdt_numerator = (p['p']*self.infiltration_factor(B, S, **kwargs)
                          - self.drainage(B, S, **kwargs)
                          - self.tree_root_function(B, S, **kwargs)
                          - self.evapotranspiration(B, S, **kwargs))
                          
        dsdt = dsdt_numerator / (p['n'] * p['zr'])
        
        return np.array([dbdt, dsdt])

    def time_integration(self, init_guess=None, t_span=None, t_eval=None, **kwargs):
        if init_guess is None:
            init_guess = self.params['initial_guess']
        if t_span is None:
            t_span = (0, self.params['t_span_max'])
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], self.params['t_eval_points'])
        
        # Use a lambda function to adapt self.equ signature (y -> [t, y]) for solve_ivp
        sol = solve_ivp(lambda t, y: self.equ(y, **kwargs), t_span, init_guess, t_eval=t_eval)
        return sol
    
    def solutions_fsolve(self, **kwargs):
        """
        Attempts to find a steady-state solution (B, S) using fsolve with 
        multiple random initial guesses, falling back to time integration.
        """
        MAX_ITER = self.params['fsolve_iterations']
        init_guess = self.params['initial_guess'] 

        # 1. Root-Finding (fsolve)
        for i in range(MAX_ITER):
            
            # The function equ is used directly because fsolve expects f(y) -> roots
            solver_func = lambda y: self.equ(y, **kwargs)
            sol_fsolve = fsolve(solver_func, init_guess)
            B, S = sol_fsolve

            # Check convergence criteria (s in (0, 1) and B > threshold (0.010 kg/m^2))
            if (B > 0.005) and (0 < S < 1):
                return [B, S]

            # If criteria failed, prepare the next random initial guess
            init_guess = self._get_random_guess()

        # 2. Time Integration Fallback (solve_ivp)
        print(f"p={self.params['p']}, fsolve failed to converge to a valid solution. Falling back to time integration.")
        
        t_span = (0, self.params['t_span_max'])
        t_eval = np.linspace(t_span[0], t_span[1], self.params['t_eval_points'])
        sol = self.time_integration(init_guess=init_guess, t_span=t_span, t_eval=t_eval, **kwargs)

        # Extract the final values (assumed equilibrium)
        B_final = sol.y[0, -1]
        S_final = sol.y[1, -1]

        return [B_final, S_final]
    
    def solutions_fast(self, **kwargs):
        """
        Attempts to find a steady-state solution (B, S) using fsolve with 
        multiple random initial guesses, falling back to time integration.
        """
        MAX_ITER = self.params['fsolve_iterations']
        init_guess = self.params['initial_guess'] 

        # 1. Root-Finding (fsolve)
        for i in range(MAX_ITER):

            # The function equ is used directly because fsolve expects f(y) -> roots
            solver_func = lambda y: self.equ(y, **kwargs)
            sol_fsolve = fsolve(solver_func, init_guess)
            B, S = sol_fsolve

            # Check convergence criteria (s in (0, 1) and B > threshold (0.010 kg/m^2))
            if (B > 0.005) and (0 < S < 1):
                return [B, S]

            # If criteria failed, prepare the next random initial guess
            init_guess = self._get_random_guess()

        return [0, 0]
