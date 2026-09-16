import pandas as pd
import numpy as np
import sympy as sp
from tqdm import tqdm
from sympy.printing import latex


def numpy_cond(x, y):
    return np.where(np.asarray(x) > 0, np.asarray(y), 0.0)


def numpy_safe_inv(x):
    x_arr = np.asarray(x)
    return np.where(x_arr == 0, 1.0, 1.0 / x_arr)


def numpy_safe_sqrt(x):
    return np.sqrt(np.maximum(0, np.asarray(x)))


ALL_SYMBOLS = sp.symbols("Vp Np Bzsouth Bmag By DST P_dyn VBs epsilon DST")
SYMBOLS_DICT = {s.name: s for s in ALL_SYMBOLS}

# Global dictionary for SymPy parsing
LOCALS_DICT = {
    "Max": sp.Max,
    "Min": sp.Min,
    "sign": sp.sign,
    "inv": lambda x: 1 / x,
    "cond": sp.Function("cond"),
    "sqrt": sp.Function("sqrt"),
    "square": lambda x: x**2,
    **SYMBOLS_DICT,
}

# Global dictionary for NumPy execution
MODULES_DICT = {
    "Max": np.maximum,
    "Min": np.minimum,
    "neg": np.negative,
    "sign": np.sign,
    "inv": numpy_safe_inv,
    "cond": numpy_cond,
    "sqrt": numpy_safe_sqrt,
    "numpy": np,
}


class EquationModel:
    def __init__(self, equation_str, feature_names, is_template=False):
        """
        equation_str: The equation in sympy_format
        feature_names: List of SW features (e.g. ['P_dyn', 'VBs'])
        is_template: Boolean indicating if the equation is a template (with #1, #2, etc.) or a direct equation with feature names
        """
        
        self.is_template = is_template
        self.equation_str = equation_str
        self.feature_names = feature_names

        if is_template:

            g_variable_names = {}
            for feature_index, feature in enumerate(feature_names):
                if feature != "DST":
                    g_variable_names[f"#{feature_index + 1}"] = feature

            d_variable_names = {"#1": "DST"}

            self.equation_str = equation_str

            self.feature_names = feature_names
            self.target_name = "DST"

            self.all_vars = (
                feature_names + [self.target_name]
                if "DST" not in feature_names
                else feature_names
            )

            g_equation = equation_str.split(";")[0].strip().split("=")[1].strip()
            d_equation = equation_str.split(";")[1].strip().split("=")[1].strip()

            for key, val in g_variable_names.items():
                g_equation = g_equation.replace(key, val)
            for key, val in d_variable_names.items():
                d_equation = d_equation.replace(key, val)

            self.g_equation_sympy, self.g_numeric_func = self._parse_equation(g_equation)
            self.d_equation_sympy, self.d_numeric_func = self._parse_equation(d_equation)
        else:
            self.feature_names = feature_names
            self.target_name = "DST"
            self.all_vars = (
                feature_names + [self.target_name]
                if "DST" not in feature_names
                else feature_names
            )

            self.expr, self.func = self._parse_equation(equation_str)  

    def _parse_equation(self, raw_str):
        expr = sp.sympify(raw_str, locals=LOCALS_DICT)
        
        return expr, sp.lambdify(
            [SYMBOLS_DICT[v] for v in self.all_vars],
            expr,
            modules=[MODULES_DICT, "numpy"],
        )

    def predict(self, data_dict):
        """Expects a dict of arrays: {'Vp': [...], 'DST': [...]}"""
        args = [data_dict[v] for v in self.all_vars]
        
        if self.is_template:
            g_results = self.g_numeric_func(*args)
            d_results = self.d_numeric_func(*args)

            return g_results + d_results
        else:
            return self.func(*args)
        
    def predict_components(self, data_dict):
        """Expects a dict of arrays: {'Vp': [...], 'DST': [...]}"""
        args = [data_dict[v] for v in self.all_vars]
        
        if self.is_template:
            g_results = self.g_numeric_func(*args)
            d_results = self.d_numeric_func(*args)

            return g_results + d_results, g_results, d_results
        else:
            return self.func(*args)

    def pred_single(self, data_list):
        """Expects a list of values in the order of self.all_vars"""

        if self.is_template:
            g_results = self.g_numeric_func(*data_list)
            d_results = self.d_numeric_func(*data_list)
            return g_results + d_results
        else:
            return self.func(*data_list)

        
    def __str__(self):
        if self.is_template:
            return f"TemplateEquationModel(equation={latex(sp.simplify(self.g_equation_sympy))}+{latex(sp.simplify(self.d_equation_sympy))}, features={self.feature_names})"
        else:
            return f"SingleEquationModel(equation={latex(sp.simplify(self.expr))}, features={self.feature_names})"

    def latex_str(self):
        if self.is_template:
            return (
                latex(sp.simplify(self.g_equation_sympy))
                + " + "
                + latex(sp.simplify(self.d_equation_sympy))
                #+ f" with features {self.feature_names}"
            )
        else:
            return latex(sp.simplify(self.expr))
        
    def sympy_format(self):
        if self.is_template:
            return f"dDST/dt = [{sp.simplify(self.g_equation_sympy)}\n + {sp.simplify(self.d_equation_sympy)}]"
        else:
            return f"dDST/dt = {sp.simplify(self.expr)}"
        
        

def simulate_storm(model, storm_df):
    """Reconstructs DST profile using Euler integration."""
    initial_idx = storm_df.index[0]
    DST_current = storm_df.loc[initial_idx, "DST"]  # Start with actual DST
    
    predictions = []
    
    # Record initial state prediction (t=0)
    initial_pred = {
        "datetime": initial_idx,
        "DST_pred": DST_current,
        "dDST": 0.0
    }
    
    # Keep schema consistent for template models at t=0
    if model.is_template:
        initial_pred["injection_component"] = 0.0
        initial_pred["decay_component"] = 0.0
        
    predictions.append(initial_pred)

    # Loop from 0 to N-2 safely
    for hour in range(len(storm_df) - 1):

        # Prepare inputs for this specific time step
        input_step = storm_df.iloc[hour].to_dict()
        input_step["DST"] = DST_current  # Crucial: Use predicted DST, not real DST

        if model.is_template:
            # Get dDST/dt
            ddst_dt, injection_component, decay_component = model.predict_components(input_step)

            # Integrate (assuming 1-hour resolution)
            DST_pred = DST_current + ddst_dt

            predictions.append(
                {
                    "datetime": storm_df.index[hour + 1],
                    "DST_pred": DST_pred,
                    "dDST": ddst_dt,
                    "injection_component": injection_component,
                    "decay_component": decay_component,
                }
            )
        else:
            # Get dDST/dt
            ddst_dt = model.predict(input_step)

            # Integrate (assuming 1-hour resolution)
            DST_pred = DST_current + ddst_dt

            predictions.append(
                {
                    "datetime": storm_df.index[hour + 1],
                    "DST_pred": DST_pred,
                    "dDST": ddst_dt,
                }
            )

        DST_current = DST_pred  # Update for next iteration

    return pd.DataFrame(predictions).set_index("datetime")


if __name__ == "__main__":
    # Example usage
    sample_eq = "g = ((inv(#4) + sqrt(#2)) * ((((3.247047 - (#1 * 0.019395135)) * #3) * 0.067984655) - -0.810041)) + -4.08069; d = square(1.4650735 - (#1 * 0.014578719)) + ((2.7653375 - #1) * cond(#1, 0.067984655))"
    feature_names = ["Vp", "Np", "Bzsouth", "Bmag", "By"]

    model = EquationModel(sample_eq, feature_names, is_template=True)

    # Mock storm data
    storm_data = pd.DataFrame(
        {
            "Vp": [1.0, 2.0, 3.0],
            "Np": [1.0, 2.0, 3.0],
            "Bzsouth": [1.0, 2.0, 3.0],
            "Bmag": [1.0, 2.0, 3.0],
            "By": [1.0, 2.0, 3.0],
            "DST": [0.1, 0.2, 0.3],
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="H"),
    )

    predictions = simulate_storm(model, storm_data)
    print(predictions)
    
    g_results = model.g_numeric_func(*[storm_data[feat].values for feat in feature_names + ["DST"]])
    d_results = model.d_numeric_func(*[storm_data[feat].values for feat in feature_names + ["DST"]])

    print(g_results + d_results)    
    
    
    sample_eq = "((-0.06690711 * DST) + -2.5263886) + ((((((Bzsouth + Bzsouth) + DST) * -0.008291428) - sqrt(Np)) + -0.60579747) * (((((Vp - Bmag) - 159.63905) * Bzsouth) * 0.0013296593) - 0.84098405))"
    feature_names = [
        "Vp",
        "Np",
        "Bzsouth",
        "Bmag",
        "DST",
    ]

    model = EquationModel(sample_eq, feature_names, is_template=False)

    # Mock storm data
    storm_data = pd.DataFrame(
        {
            "Vp": [1.0, 2.0, 3.0],
            "Np": [1.0, 2.0, 3.0],
            "Bzsouth": [1.0, 2.0, 3.0],
            "Bmag": [1.0, 2.0, 3.0],
            "DST": [0.1, 0.2, 0.3],
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="H"),
    )

    predictions = simulate_storm(model, storm_data)
    print(predictions)

    print(model.func(*[storm_data[feat].values for feat in feature_names]))
