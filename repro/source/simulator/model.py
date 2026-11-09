import os
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_model_json(model_name: str, extra_dirs=()) -> str:
    """Locate <model_name>.json, tolerating the Repressor_iFFL / Repressor_IFFL case mismatch.

    The class name is `Repressor_iFFL` but the files committed under this directory are
    spelled `Repressor_IFFL.json`. That resolves on a case-insensitive filesystem (macOS
    APFS) and fails on a case-sensitive one (Linux, and any CI container), which
    used to make every entry point here Mac-only. Search order is deliberate:
    __file__-derived locations first, so the answer no longer depends on the
    working directory either. All copies of this file are byte-identical, so which
    one is found does not affect any result.
    """
    target = model_name + '.json'
    candidates = [os.path.join(_PKG_ROOT, 'models'),
                  os.path.join(_PKG_ROOT, 'notebooks', 'models'),
                  *extra_dirs]
    for d in candidates:
        exact = os.path.join(d, target)
        if os.path.isfile(exact):
            return exact
    for d in candidates:                      # case-insensitive fallback
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.lower() == target.lower():
                return os.path.join(d, name)
    raise FileNotFoundError(
        f'could not find {target} (or a case variant) in any of: {candidates}')


class Simulator():
    """Base class to simulate the dynamics of an ODE model."""

    def __init__(self, model_name: str, t_max: float) -> None:
        """Constructor for the Simulator class.
        Args:
            model_name: The name of the model to simulate.
            t_max: The maximum time to simulate the model for.
        """
        self.model_name = model_name
        self.t_max = t_max
        self._base_path = os.path.abspath('')
        self._load_model()

    def _load_model(self) -> None:
        """Load the model parameters and initial conditions."""
        self._load_default_parameters()
        self._load_default_initial_conditions()

    def _load_default_parameters(self) -> None:
        """Load the default model parameters."""
        # Open the JSON file
        with open(_resolve_model_json(
                self.model_name,
                extra_dirs=(os.path.join(self._base_path, 'models'),))) as f:
            data = json.load(f)
        # Load the parameters
        self.params = data['params']

    def _load_default_initial_conditions(self) -> None:
        """Load the default initial conditions."""
        # Open the JSON file
        with open(_resolve_model_json(
                self.model_name,
                extra_dirs=(os.path.join(self._base_path, 'models'),))) as f:
            data = json.load(f)
        # Load the parameters
        self.initial_conditions = data['initial_conditions']
        self.state = tuple(self.initial_conditions.keys())

    def _adjust_parameters(self) -> None:
        """Adjust the model parameters."""
        raise NotImplementedError

    def _set_x0(self) -> None:
        """Set the initial state vector."""
        self.x0 = np.array([self.initial_conditions[s] for s in self.state])

    def update_parameters(self, params: dict) -> dict:
        """Update the model parameters.
        Args:
            params: The model parameters.
        Returns:
            The updated parameters.
        """
        self.params.update(params)
        return self.params

    def update_initial_conditions(self, initial_conditions: dict) -> dict:
        """Update the initial conditions.
        Args:
            initial_conditions: The initial conditions for the state vector.
        Returns:
            The updated initial conditions.
        """
        self.initial_conditions.update(initial_conditions)
        return self.initial_conditions

    def model(self, t: float, x: np.ndarray) -> np.ndarray:
        """The ODE model.
        Args:
            t: The time.
            x: The state vector.
        Returns:
            The derivative of the state vector.
        """
        raise NotImplementedError

    def simulate_model(self) -> pd.DataFrame:
        "Simulate the ODE model."
        # Adjust the model parameters
        self._adjust_parameters()
        # Set the initial conditions
        self._set_x0()
        # Simulate the model
        tspan = (0., self.t_max)
        t_eval = np.arange(tspan[0], tspan[1]+1.)
        soln = solve_ivp(self.model, tspan, self.x0, method='LSODA', t_eval=t_eval, rtol=1e-6, atol=1e-9)
        # Store the solution
        self.t = {'t': soln.t}
        self.x = {key: value for key, value in zip(self.state, soln.y)}
        return pd.DataFrame(self.t | self.x)

class Repressor_iFFL(Simulator):
    """"Cass to simulate the DNA- or RNA-endoded iFFL architecture with Repressor in either 5'UTR or 3'UTR."""

    def __init__(self, t_max: float, circuit_type: str, repression_binding_site: str, control_flag: bool = False) -> None:
        """Constructor for the Base_Repressor_iFFL class.
        Args:
            t_max: The maximum time to simulate the model for.
            circuit_type: The circuit type (DNA or RNA).
            repression_binding_site: The position of the Repressor binding site.
            control_flag: Whether to simulate the biological control (no Repressor binding sites).
        """
        model_name = self.__class__.__name__
        self.circuit_type = circuit_type
        self.repression_binding_site = repression_binding_site
        self.control_flag = control_flag
        self.model_flags = {'isbranch_1': False, 'isbranch_2': False}
        self.x0 = None # The initial state vector is set to None
        match self.repression_binding_site:
            case 'branch_1':
                self.model_flags = {'isbranch_1': True, 'isbranch_2': False}
            case 'branch_2':
                self.model_flags = {'isbranch_1': False, 'isbranch_2': True}
            case _:
                raise ValueError('Invalid value for the Repressor binding site.')
        match self.circuit_type:
            case 'DNA':
                self.model = self.DNA_model
            case 'RNA':
                self.model = self.RNA_model
            case _:
                raise ValueError('Invalid value for the circuit type.')
        # Initialise the base class
        super().__init__(model_name, t_max)

    def _load_default_initial_conditions(self) -> None:
        """Load the default initial conditions."""
        # Open the JSON file
        with open(_resolve_model_json(
                self.model_name,
                extra_dirs=(os.path.join(self._base_path, 'models'),))) as f:
            data = json.load(f)
        # Load the parameters
        self.initial_conditions = data['initial_conditions_' + self.circuit_type]
        self.state = tuple(self.initial_conditions.keys())

    def _adjust_parameters(self) -> None:
        """Adjust the model parameters."""
        length_repression = self.params['length_model_1'] * self.model_flags['isbranch_1'] + self.params['length_model_2'] * self.model_flags['isbranch_2']
        aux_params = {
            'alpha_output': self.params['alpha'] / self.params['length_output'] / 3,
            'alpha_repression': self.params['alpha'] / length_repression / 3,
            'alpha_burden': self.params['alpha'] / self.params['length_burden'] / 3,
            'gamma_output': self.params['gamma'] / self.params['length_output'],
            'gamma_repression': self.params['gamma'] / length_repression,
            'gamma_burden': self.params['gamma'] / self.params['length_burden'],
            'phi_output': self.params['length_output'] / self.params['ribo_footprint_size'],
            'phi_repression': length_repression / self.params['ribo_footprint_size'],
            'phi_burden': self.params['length_burden'] / self.params['ribo_footprint_size']
            }
        self.params.update(aux_params)
        aux_params = {
            'nu_output': (self.params['alpha_output'] + self.params['nu_minus']) / self.params['nu_plus'],
            'nu_repression': (self.params['alpha_repression'] + self.params['nu_minus']) / self.params['nu_plus'],
            'nu_burden': (self.params['alpha_burden'] + self.params['nu_minus']) / self.params['nu_plus'],
            'kappa_output': (self.params['gamma_output'] + self.params['kappa_minus']) / self.params['kappa_plus'] / self.params['phi_output'],
            'kappa_repression': (self.params['gamma_repression'] + self.params['kappa_minus']) / self.params['kappa_plus'] / self.params['phi_repression'],
            'kappa_burden': (self.params['gamma_burden'] + self.params['kappa_minus']) / self.params['kappa_plus']  / self.params['phi_burden'],
            'diff_beta': self.params['beta'] * self.params['lambda_beta'] * self.model_flags['isbranch_2'],
            'theta': self.params['theta'] * self.model_flags['isbranch_2']
            }
        # Adjust the parameter eta_plus if control_flag is true
        if self.control_flag:
            aux_params['eta_plus'] = 0.
            aux_params['eta_minus'] = 0.
        else:
            aux_params['eta_plus'] = (
                self.params['eta_plus_model_1'] * self.model_flags['isbranch_1'] + 
                self.params['eta_plus_model_2'] * self.model_flags['isbranch_2']
                )
            aux_params['eta_minus'] = (
                self.params['eta_minus_model_1'] * self.model_flags['isbranch_1'] + 
                self.params['eta_minus_model_2'] * self.model_flags['isbranch_2']
                )
        # Update the parameters
        self.params.update(aux_params)

    def RNA_model(self, t: float, x: np.ndarray) -> np.ndarray:
        """The ODE model for RNA-encoded iFFL with Repressor in either 5'UTR or 3'UTR.
        Args:
            t: The time.
            x: The state vector.
        Returns:
            The derivative of the state vector.
        """
        # Get the model parameters
        params = self.params
        # Unpack the state vector
        RNA_Output, RNA_Output_Repressor, RNA_Repressor, RNA_Burden, Output, Repressor, Burden = x
        # Calculate the total RNA for Output
        RNA_Output_total = RNA_Output + params['theta'] * RNA_Output_Repressor 
        # Calculate the free ribosome concentration
        free_RTL = params['total_rtl'] / (
            1 + (RNA_Output_total / params['kappa_output'] + RNA_Repressor / params['kappa_repression'] + RNA_Burden / params['kappa_burden']) 
            )
        # Calculate the derivatives
        d_RNA_Output = params['eta_minus'] * RNA_Output_Repressor - (params['beta'] + params['eta_plus'] * Repressor) * RNA_Output
        d_RNA_Output_Repressor = params['eta_plus'] * Repressor * RNA_Output - (params['beta'] + params['diff_beta'] + params['eta_minus']) * RNA_Output_Repressor
        d_RNA_Repressor = - params['beta']  * RNA_Repressor
        d_RNA_Burden = - params['beta'] * RNA_Burden
        d_Output = params['gamma_output'] * RNA_Output_total / params['kappa_output'] * free_RTL - (params['delta'] + params['mu']) * Output
        d_Repressor = (
            params['gamma_repression'] * RNA_Repressor / params['kappa_repression'] * free_RTL + 
            (params['beta'] + params['diff_beta'] + params['eta_minus']) * RNA_Output_Repressor - 
            (params['delta'] + params['mu'] + params['eta_plus'] * RNA_Output) * Repressor
            )
        d_Burden = params['gamma_burden'] * RNA_Burden / params['kappa_burden'] * free_RTL - (params['delta'] + params['mu']) * Burden
        # Return the derivatives
        return np.array([
            d_RNA_Output, 
            d_RNA_Output_Repressor, 
            d_RNA_Repressor, 
            d_RNA_Burden, 
            d_Output, 
            d_Repressor, 
            d_Burden
            ])

    def DNA_model(self, t: float, x: np.ndarray) -> np.ndarray:
        """The ODE model for DNA-encoded iFFL with Repressor in either 5'UTR or 3'UTR.
        Args:
            t: The time.
            x: The state vector.
        Returns:
            The derivative of the state vector.
        """
        # Get the model parameters
        params = self.params
        # Unpack the state vector
        DNA_Output, DNA_Repressor, DNA_Burden, RNA_Output, RNA_Output_Repressor, RNA_Repressor, RNA_Burden, Output, Repressor, Burden = x
        # Calculate the total RNA for Output
        RNA_Output_total = RNA_Output + params['theta'] * RNA_Output_Repressor  
        # Calculate the free RNA polymerase concentration
        free_RTX = params['total_rtx'] / (
            1 + (DNA_Output / params['nu_output'] + DNA_Repressor / params['nu_repression'] + DNA_Burden / params['nu_burden'])
            )
        # Calculate the free ribosome concentration
        free_RTL = params['total_rtl'] / (
            1 + (RNA_Output_total / params['kappa_output'] + RNA_Repressor / params['kappa_repression'] + RNA_Burden / params['kappa_burden'])
            )
        # Calculate the derivatives
        d_DNA_Output = - params['mu'] * DNA_Output
        d_DNA_Repressor = - params['mu'] * DNA_Repressor
        d_DNA_Burden = - params['mu'] * DNA_Burden
        d_RNA_Output = (params['alpha_output'] * DNA_Output / params['nu_output'] * free_RTX + 
                      params['eta_minus'] * RNA_Output_Repressor - 
                      (params['beta'] + params['eta_plus'] * Repressor) * RNA_Output
                      )
        d_RNA_Output_Repressor = params['eta_plus'] * Repressor * RNA_Output - (params['beta'] + params['diff_beta'] + params['eta_minus']) * RNA_Output_Repressor
        d_RNA_Repressor = (params['alpha_repression'] * DNA_Repressor / params['nu_repression'] * free_RTX -
            params['beta'] * RNA_Repressor
            )
        d_RNA_Burden = (
            params['alpha_burden'] * DNA_Burden / params['nu_burden'] * free_RTX -
            params['beta'] * RNA_Burden)
        d_Output = params['gamma_output'] * RNA_Output_total / params['kappa_output'] * free_RTL - (params['delta'] + params['mu']) * Output
        d_Repressor = (
            params['gamma_repression'] * RNA_Repressor / params['kappa_repression'] * free_RTL + 
            (params['beta'] + params['diff_beta'] + params['eta_minus']) * RNA_Output_Repressor - 
            (params['delta'] + params['mu']+ params['eta_plus'] * RNA_Output) * Repressor
            )
        d_Burden = params['gamma_burden'] * RNA_Burden / params['kappa_burden'] * free_RTL - (params['delta'] + params['mu']) * Burden
        # Return the derivatives
        return np.array([
            d_DNA_Output, 
            d_DNA_Repressor, 
            d_DNA_Burden, 
            d_RNA_Output, 
            d_RNA_Output_Repressor, 
            d_RNA_Repressor, 
            d_RNA_Burden, 
            d_Output, 
            d_Repressor, 
            d_Burden
            ])
