"""Physics subpackage: rheology model fitting and dataset construction.

Modules
-------
models.py
    Implements five shear-thinning viscosity models (Power Law, Sisko,
    Williamson, Carreau-Yasuda CY5, Constrained CY) as scipy least-squares
    fitters.  The ``FITTERS`` registry maps config keys to callables.

datasets.py
    Uses the fitters to produce per-row physics parameter datasets from raw
    viscosity measurements.  The fitted parameters become the regression
    targets for the XGBoost models.
"""
