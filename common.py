"""Shared constants, classes, and helpers for the PINN pipeline.
"""

import copy
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Version-agnostic trapezoidal integration: NumPy < 2.0 has np.trapz,
# NumPy >= 2.0 renamed it to np.trapezoid (and dropped np.trapz in some
# 2.x patch releases). Use this shim everywhere instead of np.trapz /
# np.trapezoid so the notebook runs on either NumPy generation.
np_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
if np_trapz is None:
    raise ImportError("Neither np.trapezoid nor np.trapz is available "
                      "in this NumPy install; please upgrade NumPy.")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")
torch.set_num_threads(2)


# -----------------------------------------------------------------------
# Physical / numerical constants
# -----------------------------------------------------------------------
Z_MAX = 0.30        # m, depth of the 1-D sediment column
Z_SENSOR = 0.02     # m, depth of the buried sediment-temperature sensor

# Bounds for the learnable parameters (sigmoid-remapped)
KAPPA_MIN, KAPPA_MAX = 1.0e-7, 1.0e-6     # m^2 s^-1
KAPPA_INIT = 5.0e-7
ALPHA_MIN, ALPHA_MAX = -5.0e-1, 5.0e-1     # K^-1 (signed: positive = thermal expansion;
                                            # negative = thermal consolidation, the
                                            # dominant process in organic, water-
                                            # saturated pond sediments per Delage
                                            # et al. 2000)
ALPHA_INIT = -1.0e-2                        # start with a small negative value

# Loss weights (Eq. 12)
LAMBDA_PDE = 1.0
LAMBDA_LT = 10.0
LAMBDA_GNSS = 20.0
LAMBDA_BC = 10.0
LAMBDA_IC = 5.0


# -----------------------------------------------------------------------
# Data ingestion + dimensional rescaling of the surface forcing
# -----------------------------------------------------------------------
def load_data(data_dir="."):
    """Load and quality-control the three input files.

    Returns
    -------
    lt_df : DataFrame with columns
        date, delta_z_lt, S_t, T_LT_C, T_REF_C
        - S_t: cumulative LT proxy in [0, 1] (or unsigned)
        - T_LT_C: surface temperature time series in degrees Celsius
                 reconstructed from the supplied amplitude/phase data
        - T_REF_C: column-mean reference temperature used in the
                  thermoelastic depth integral
    gnss_df : DataFrame, 104 points with (easting, northing, dz_gnss)
    """
    # 1) Daily LT delta-z series (interpolation to a uniform daily grid).
    lt_df = (pd.read_excel(os.path.join(data_dir, "cleaned_lt_daily_delta_z.xlsx"),
                           parse_dates=["date"])
             .sort_values("date").reset_index(drop=True))

    # honest gap count (C9 fix)
    full_index = pd.date_range(lt_df["date"].min(),
                               lt_df["date"].max(), freq="D")
    n_gaps = (~lt_df["date"].isin(full_index)).sum() + \
             (len(full_index) - len(lt_df))
    lt_df = (lt_df.set_index("date").reindex(full_index)
             .rename_axis("date").interpolate(method="linear")
             .reset_index())

    # 2) Cumulative LT proxy S(t) (the "shape" of the seasonal forcing).
    lt_df["cumulative_lt"] = lt_df["delta_z_lt"].cumsum()
    cum_t2 = lt_df["cumulative_lt"].iloc[-1]
    if abs(cum_t2) < 1e-12:
        raise ValueError("Cumulative LT at t2 is effectively zero.")
    lt_df["S_t"] = lt_df["cumulative_lt"] / cum_t2

    # 3) Reconstruct an actual surface-temperature time series T_LT(t) in
    #    degrees Celsius. The LT amplitude/phase file gives daily diurnal
    #    amplitudes of the water column (Aw) and sediment surface (As)
    #    plus phases. Following the review's recommendation, we anchor
    #    the seasonal envelope of T_LT(t) to a plausible British pond
    #    range and modulate it by S(t):
    #
    #        T_LT(t) = T_BASE + DELTA_T * S_eff(t)
    #
    #    where T_BASE = 4 degC (typical winter water temperature),
    #          DELTA_T = 18 K (giving summer maxima ~ 22 degC),
    #          S_eff(t) = (S(t) - S_min) / (S_max - S_min) maps the
    #          cumulative proxy onto the unit interval whatever its sign.
    lt_ap_path = os.path.join(data_dir, "LT_amplitude_and_phase.xlsx")
    has_ap = os.path.exists(lt_ap_path)
    if has_ap:
        # cross-check: amplitude data should span the same period
        lt_ap = pd.read_excel(lt_ap_path, parse_dates=["date"])
        # store for diagnostics (not used in the boundary condition)
        lt_df = lt_df.merge(lt_ap, on="date", how="left")

    T_BASE_C = 4.0
    DELTA_T_C = 18.0
    s = lt_df["S_t"].values.astype(np.float32)
    s_eff = (s - s.min()) / (s.max() - s.min() + 1e-12)
    lt_df["T_LT_C"] = T_BASE_C + DELTA_T_C * s_eff

    # Reference temperature for the thermoelastic depth integral: the
    # initial-time column-mean. With an IC of T(z, 0) = T_LT(0), the
    # mean equals T_LT(0), so we use that.
    lt_df["T_REF_C"] = float(lt_df["T_LT_C"].iloc[0])

    # 4) GNSS surveys.
    gnss_df = pd.read_excel(os.path.join(data_dir, "GNSS.xlsx"))
    gnss_df.columns = gnss_df.columns.str.strip()
    return lt_df, gnss_df, n_gaps


# -----------------------------------------------------------------------
# Pure-NumPy Latin Hypercube Sampling
# -----------------------------------------------------------------------
def lhs(n_samples, n_dim, rng):
    seg = np.linspace(0, 1, n_samples + 1)
    pts = rng.uniform(seg[:-1], seg[1:], size=(n_dim, n_samples)).T
    for j in range(n_dim):
        rng.shuffle(pts[:, j])
    return pts.astype(np.float32)


# -----------------------------------------------------------------------
# PINN model
# -----------------------------------------------------------------------
class PINN(nn.Module):
    """4-layer 32-neuron feedforward network with two learnable scalars.

    The network outputs temperature in degrees Celsius. The learnable
    scalars (kappa, alpha) are constrained to physical ranges via sigmoid
    remapping.
    """

    def __init__(self, n_layers=4, n_neurons=32, z_max=Z_MAX,
                 kappa_init=KAPPA_INIT, alpha_init=ALPHA_INIT):
        super().__init__()
        self.z_max = z_max
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Robust clip: keep init strictly inside (min, max) regardless of sign.
        eps_k = 0.01 * (KAPPA_MAX - KAPPA_MIN)
        eps_a = 0.01 * (ALPHA_MAX - ALPHA_MIN)
        ki = float(np.clip(kappa_init, KAPPA_MIN + eps_k, KAPPA_MAX - eps_k))
        ai = float(np.clip(alpha_init, ALPHA_MIN + eps_a, ALPHA_MAX - eps_a))
        self.kappa_raw = nn.Parameter(torch.tensor(
            float(np.log((ki - KAPPA_MIN) / (KAPPA_MAX - ki))),
            dtype=torch.float32))
        self.alpha_raw = nn.Parameter(torch.tensor(
            float(np.log((ai - ALPHA_MIN) / (ALPHA_MAX - ai))),
            dtype=torch.float32))

    @property
    def kappa(self):
        return KAPPA_MIN + (KAPPA_MAX - KAPPA_MIN) * torch.sigmoid(self.kappa_raw)

    @property
    def alpha(self):
        return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * torch.sigmoid(self.alpha_raw)

    def forward(self, z, t):
        return self.net(torch.cat([z / self.z_max, t], dim=-1))


# Data-only baseline (no PDE / BC / IC losses; same architecture).
class DataMLP(nn.Module):
    def __init__(self, n_layers=4, n_neurons=32, z_max=Z_MAX,
                 alpha_init=ALPHA_INIT):
        super().__init__()
        self.z_max = z_max
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        eps_a = 0.01 * (ALPHA_MAX - ALPHA_MIN)
        ai = float(np.clip(alpha_init, ALPHA_MIN + eps_a, ALPHA_MAX - eps_a))
        self.alpha_raw = nn.Parameter(torch.tensor(
            float(np.log((ai - ALPHA_MIN) / (ALPHA_MAX - ai))),
            dtype=torch.float32))

    @property
    def alpha(self):
        return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * torch.sigmoid(self.alpha_raw)

    def forward(self, z, t):
        return self.net(torch.cat([z / self.z_max, t], dim=-1))


# -----------------------------------------------------------------------
# Losses (operate in degrees Celsius)
# -----------------------------------------------------------------------
def pde_residual(model, z_c, t_c, period_days):
    """Physics residual of the 1-D heat equation in physical units.

    The network's time coordinate is normalised to [0, 1] for stability,
    so we scale the time derivative back to physical days:
        dT/dt_physical = (1 / period_days) * dT/dt_norm
    and use kappa in m^2 day^-1 throughout (kappa_day = kappa * 86400).
    """
    z_c = z_c.requires_grad_(True)
    t_c = t_c.requires_grad_(True)
    T = model(z_c, t_c)
    dT_dt = torch.autograd.grad(T, t_c, torch.ones_like(T), create_graph=True)[0]
    dT_dz = torch.autograd.grad(T, z_c, torch.ones_like(T), create_graph=True)[0]
    d2T_dz2 = torch.autograd.grad(dT_dz, z_c, torch.ones_like(dT_dz),
                                  create_graph=True)[0]
    kappa_day = model.kappa * 86400.0
    return torch.mean((dT_dt / period_days - kappa_day * d2T_dz2) ** 2)


def thermo_u(model, t_scalar, T_REF, n_z=100):
    """Vertical displacement via the standard thermoelastic kernel.

    u(t) = alpha * integral over z of (T(z, t) - T_REF) dz

    (C5 fix: no explicit z-weighting; T_REF is the initial-time column
    mean temperature, passed in from data.)
    """
    z_i = torch.linspace(0, model.z_max, n_z, device=DEVICE).unsqueeze(1)
    t_i = torch.full_like(z_i, t_scalar)
    T_i = model(z_i, t_i)
    return torch.trapz((model.alpha * (T_i - T_REF)).squeeze(),
                       dx=model.z_max / (n_z - 1))


def total_loss(model, z_c, t_c, t_lt, T_LT_obs, u1, u2, t_all,
               T_REF, period_days, include_pde_bc_ic=True):
    """Composite loss in degrees Celsius / metres.

    Parameters
    ----------
    T_LT_obs : tensor of surface temperatures (degC) at observation times
    T_REF    : scalar (degC), reference temperature for the depth integral
    period_days : physical duration of the monitoring window (days)

    include_pde_bc_ic : if False (data-only baseline B3) the PDE/BC/IC
        terms are omitted and only the LT-surface + GNSS data losses
        are minimised.
    """
    # LT surface: enforce T_theta(0, t) = T_LT(t) in degC
    L_lt = torch.mean((model(torch.zeros_like(t_lt), t_lt) - T_LT_obs) ** 2)

    # GNSS amplitude (Eq. 11)
    u_t1 = thermo_u(model, 0.0, T_REF)
    u_t2 = thermo_u(model, 1.0, T_REF)
    u1_t = torch.tensor(u1, dtype=torch.float32, device=DEVICE)
    u2_t = torch.tensor(u2, dtype=torch.float32, device=DEVICE)
    L_gnss = (u_t1 - u1_t) ** 2 + (u_t2 - u2_t) ** 2

    if not include_pde_bc_ic:
        L_tot = LAMBDA_LT * L_lt + LAMBDA_GNSS * L_gnss
        return {"total": L_tot, "PDE": 0.0, "LT": L_lt.item(),
                "GNSS": L_gnss.item(), "BC": 0.0, "IC": 0.0}

    # PDE residual
    L_pde = pde_residual(model, z_c, t_c, period_days)

    # Base boundary: at z = z_max we expect a quasi-stationary temperature
    # equal to the seasonal mean of the surface forcing (not an arbitrary
    # 12 degC value as in V4). Use the time-mean of T_LT_obs.
    T_base = torch.mean(T_LT_obs)
    z_b = torch.full_like(t_all, model.z_max)
    L_bc = torch.mean((model(z_b, t_all) - T_base) ** 2)

    # Initial condition: column starts uniform at T_LT(0)
    z_i = torch.linspace(0, model.z_max, 50, device=DEVICE).unsqueeze(1)
    t_i = torch.zeros_like(z_i)
    T0 = T_LT_obs[0].item()
    L_ic = torch.mean((model(z_i, t_i) - T0) ** 2)

    L_tot = (LAMBDA_PDE * L_pde + LAMBDA_LT * L_lt + LAMBDA_GNSS * L_gnss
             + LAMBDA_BC * L_bc + LAMBDA_IC * L_ic)
    return {"total": L_tot,
            "PDE": L_pde.item(), "LT": L_lt.item(), "GNSS": L_gnss.item(),
            "BC": L_bc.item(), "IC": L_ic.item()}
