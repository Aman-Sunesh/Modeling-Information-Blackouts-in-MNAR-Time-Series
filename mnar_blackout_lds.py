# ============================================================
# Module: mnar_blackout_lds
# ------------------------------------------------------------
# - MNAR-aware linear dynamical system for traffic blackouts
# - Extended Kalman filter using speeds x_t and masks m_t
# - Rauch–Tung–Striebel (RTS) smoother
# - Utilities for reconstruction and k-step forecasting
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    # Clip inputs to avoid overflow in exp for large |x|
    x_clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


# ------------------------------------------------------------
# Parameter container
# ------------------------------------------------------------

@dataclass
class MNARParams:
    """
    Container for MNAR LDS parameters.

    Dimensions
    ----------
    - K : latent state dimension
    - D : number of detectors / observed variables
    """
    A: np.ndarray          # (K, K)  state transition
    Q: np.ndarray          # (K, K)  process noise covariance
    C: np.ndarray          # (D, K)  emission matrix
    R: np.ndarray          # (D, D)  observation noise covariance for speeds
    mu0: np.ndarray        # (K,)    initial mean
    Sigma0: np.ndarray     # (K, K)  initial covariance
    phi: np.ndarray        # (D, K)  MNAR logistic parameters per detector

    @property
    def K(self) -> int:
        # Latent state dimension K (dimensionality of z_t)
        return self.A.shape[0]

    @property
    def D(self) -> int:
        # Observation dimension D (number of detectors / observed speeds)
        return self.C.shape[0]

    @staticmethod
    def init_random(K: int, D: int, seed: int = 0) -> "MNARParams":
        """
        Simple random initialization for debugging / experimentation.
        Replace with learned params for real experiments.
        """
        rng = np.random.default_rng(seed)

        # Randomly stable-ish A (shrink towards identity)
        A = np.eye(K) + 0.05 * rng.standard_normal((K, K))
        # Process noise (diagonal for simplicity)
        Q_diag = 0.1 * np.ones(K, dtype=float)
        Q = np.diag(Q_diag)

        # Emission matrix C: map K-dim latent state to D detectors
        C = 0.1 * rng.standard_normal((D, K))

        # Observation noise for speeds (diagonal)
        R_diag = 4.0 * np.ones(D, dtype=float)  # e.g. ~2 m/s std
        R = np.diag(R_diag)

        # Initial state prior
        mu0 = np.zeros(K, dtype=float)
        Sigma0 = np.eye(K, dtype=float)

        # MNAR logistic parameters per detector
        phi = 0.1 * rng.standard_normal((D, K))

        return MNARParams(A=A, Q=Q, C=C, R=R, mu0=mu0, Sigma0=Sigma0, phi=phi)


# ------------------------------------------------------------
# Core MNAR-aware EKF
# ------------------------------------------------------------

class MNARBlackoutLDS:
    """
    MNAR-aware linear dynamical system for traffic blackouts.

    The model matches the LaTeX description:

        z_t | z_{t-1} ~ N(A z_{t-1}, Q)
        x_t | z_t     ~ N(C z_t, R)
        m_{t,d} | z_t ~ Bernoulli( sigma( phi_d^T z_t ) )

    where:
        - x_t in R^D : detector speeds
        - m_t in {0,1}^D, with 1 = missing, 0 = observed

    Inference is done with an EKF-style update that uses:
        - observed speeds x_t (only where m_t == 0),
        - full masks m_t as an additional "pseudo-observation" block.
    """

    def __init__(self, params: MNARParams):
        self.params = params

    # --------------------------------------------------------
    # Forward pass: MNAR-aware extended Kalman filter
    # --------------------------------------------------------

    def ekf_forward(
        self,
        x_t: np.ndarray,
        m_t: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run MNAR-aware EKF over the entire sequence.

        Parameters
        ----------
        x_t : np.ndarray, shape (T, D)
            Speed panel. NaNs are allowed; they should be consistent
            with m_t (i.e., NaN when m_t[t, d] == 1).
        m_t : np.ndarray, shape (T, D), dtype in {0,1}
            Missingness masks (1 = missing, 0 = observed).

        Returns
        -------
        results : dict of np.ndarray
            - 'mu_pred'   : (T, K)  predicted means   E[z_t | y_{1:t-1}]
            - 'Sigma_pred': (T, K, K) predicted covs
            - 'mu_filt'   : (T, K)  filtered means    E[z_t | y_{1:t}]
            - 'Sigma_filt': (T, K, K) filtered covs
        """
        params = self.params
        A, Q, C, R, mu0, Sigma0, phi = (
            params.A,
            params.Q,
            params.C,
            params.R,
            params.mu0,
            params.Sigma0,
            params.phi,
        )

        T, D = x_t.shape
        K = params.K

        # Allocate arrays for predicted / filtered statistics
        mu_pred = np.zeros((T, K), dtype=float)
        Sigma_pred = np.zeros((T, K, K), dtype=float)
        mu_filt = np.zeros((T, K), dtype=float)
        Sigma_filt = np.zeros((T, K, K), dtype=float)

        # Start from prior at t = 0
        mu_prev = mu0.copy()
        Sigma_prev = Sigma0.copy()

        I_K = np.eye(K, dtype=float)

        for t in range(T):
            # ------------------------------------------------
            # 1) Predict step (standard LDS)
            # ------------------------------------------------
            mu_t_pred = A @ mu_prev
            Sigma_t_pred = A @ Sigma_prev @ A.T + Q

            # Save prediction before seeing time t data
            mu_pred[t] = mu_t_pred
            Sigma_pred[t] = Sigma_t_pred

            # ------------------------------------------------
            # 2) Build observation blocks at time t
            # ------------------------------------------------
            mask_row = m_t[t]  # shape (D,)
            # Indices of observed speeds (m = 0)
            observed_idx = np.where(mask_row == 0)[0]
            has_speeds = observed_idx.size > 0

            # --- Speed block (only if some speeds observed) ---
            if has_speeds:
                # Observed speed values y_t^{(x)}
                y_x = x_t[t, observed_idx].astype(float)  # (|O_t|,)

                # Observation model h^{(x)}(z) = C_x z
                C_x = C[observed_idx, :]                 # (|O_t|, K)
                h_x = C_x @ mu_t_pred                    # (|O_t|,)

                # Jacobian for speed block is just C_x (linear)
                J_x = C_x

                # Submatrix of R restricted to observed detectors
                R_x = R[np.ix_(observed_idx, observed_idx)]  # (|O_t|, |O_t|)
            else:
                # Placeholders (unused if has_speeds is False)
                y_x = None
                h_x = None
                J_x = None
                R_x = None

            # --- Missingness block (always present) ---
            # We model probability of m_{t,d} = 1 (missing) as:
            #   pi_{t,d} = sigma( phi_d^T mu_{t|t-1} )
            # where phi has shape (D, K) and mu_t_pred has shape (K,).
            u_m = phi @ mu_t_pred                        # (D,)
            pi = _sigmoid(u_m)                           # (D,) = P(m=1)
            # Gradient of sigmoid wrt z:  pi*(1-pi) * phi_d
            # Shape: (D, K)
            g = (pi * (1.0 - pi))[:, None] * phi

            # Treat m_t as approximate Gaussian observation:
            #   m_t ≈ pi + g (z_t - mu_t_pred) + epsilon
            # with Var(epsilon_d) ≈ pi_d (1-pi_d).
            y_m = mask_row.astype(float)                 # (D,)
            h_m = pi                                     # (D,)
            J_m = g                                      # (D, K)

            # Diagonal covariance for missingness pseudo-observations
            var_m = pi * (1.0 - pi)                      # (D,)
            # Add small jitter for numerical stability
            var_m = var_m + 1e-6
            S_m = np.diag(var_m)                         # (D, D)

            # ------------------------------------------------
            # 3) Combine blocks and run EKF update
            # ------------------------------------------------
            if has_speeds:
                # Stack observed speeds and missingness:
                # y = [ y^{(x)} ; y^{(m)} ]
                y = np.concatenate([y_x, y_m], axis=0)   # (|O_t| + D,)

                # h = [ h^{(x)} ; h^{(m)} ]
                h = np.concatenate([h_x, h_m], axis=0)   # (|O_t| + D,)

                # J = [ J^{(x)} ; J^{(m)} ]
                J = np.vstack([J_x, J_m])                # (|O_t| + D, K)

                # Block-diagonal covariance:
                #   R_t = diag(R_x, S_m)
                top_left = R_x
                top_right = np.zeros((R_x.shape[0], D), dtype=float)
                bottom_left = np.zeros((D, R_x.shape[0]), dtype=float)
                bottom_right = S_m
                R_t = np.block([
                    [top_left,    top_right],
                    [bottom_left, bottom_right],
                ])
            else:
                # No speed observations: only missingness block
                y = y_m                                   # (D,)
                h = h_m                                   # (D,)
                J = J_m                                   # (D, K)
                R_t = S_m                                 # (D, D)

            # Innovation covariance: S_y = J Σ J^T + R
            S_y = J @ Sigma_t_pred @ J.T + R_t

            # Kalman gain: K_t = Σ J^T S_y^{-1}
            K_t = Sigma_t_pred @ J.T @ np.linalg.inv(S_y)

            # Innovation: (y - h(z_pred))
            innov = y - h

            # Filtered state mean and covariance
            mu_t_filt = mu_t_pred + K_t @ innov
            Sigma_t_filt = (I_K - K_t @ J) @ Sigma_t_pred

            # Save filtered posterior
            mu_filt[t] = mu_t_filt
            Sigma_filt[t] = Sigma_t_filt

            # Prepare for next time step
            mu_prev = mu_t_filt
            Sigma_prev = Sigma_t_filt

        return {
            "mu_pred": mu_pred,
            "Sigma_pred": Sigma_pred,
            "mu_filt": mu_filt,
            "Sigma_filt": Sigma_filt,
        }

    # --------------------------------------------------------
    # Backward pass: Rauch–Tung–Striebel smoother
    # --------------------------------------------------------

    def rts_smoother(
        self,
        ekf_results: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        RTS smoother for the MNAR LDS.

        This uses the standard LDS RTS equations, applied to the
        EKF-filtered sequence (we ignore higher-order corrections and
        treat the linearization as fixed).

        Parameters
        ----------
        ekf_results : dict
            Output of ekf_forward():
            - 'mu_pred', 'Sigma_pred', 'mu_filt', 'Sigma_filt'.

        Returns
        -------
        results : dict of np.ndarray
            - 'mu_smooth'   : (T, K)   smoothed means
            - 'Sigma_smooth': (T, K, K) smoothed covariances
        """
        A = self.params.A

        mu_pred = ekf_results["mu_pred"]      # (T, K)
        Sigma_pred = ekf_results["Sigma_pred"]  # (T, K, K)
        mu_filt = ekf_results["mu_filt"]      # (T, K)
        Sigma_filt = ekf_results["Sigma_filt"]  # (T, K, K)

        T, K = mu_filt.shape

        # Initialize smoothed arrays with filtered values
        mu_smooth = mu_filt.copy()
        Sigma_smooth = Sigma_filt.copy()

        # Iterate backward: t = T-2, ..., 0
        for t in range(T - 2, -1, -1):
            # Smoother gain:
            #   F_t = Σ_{t|t} A^T (Σ_{t+1|t})^{-1}
            Sigma_f = Sigma_filt[t]            # (K, K)
            Sigma_pred_next = Sigma_pred[t + 1]  # (K, K)

            F_t = Sigma_f @ A.T @ np.linalg.inv(Sigma_pred_next)

            # Mean update:
            #   μ_{t|T} = μ_{t|t} + F_t (μ_{t+1|T} - μ_{t+1|t})
            mu_smooth[t] = (
                mu_filt[t]
                + F_t @ (mu_smooth[t + 1] - mu_pred[t + 1])
            )

            # Covariance update:
            #   Σ_{t|T} = Σ_{t|t} + F_t (Σ_{t+1|T} - Σ_{t+1|t}) F_t^T
            Sigma_smooth[t] = (
                Sigma_f
                + F_t @ (Sigma_smooth[t + 1] - Sigma_pred_next) @ F_t.T
            )

        return {
            "mu_smooth": mu_smooth,
            "Sigma_smooth": Sigma_smooth,
        }

    # --------------------------------------------------------
    # EM training for MNAR LDS
    # --------------------------------------------------------

    def em_train(
        self,
        x_t: np.ndarray,
        m_t: np.ndarray,
        num_iters: int = 5,
        update_phi: bool = True,
        phi_steps: int = 5,
        phi_lr: float = 1e-3,
        verbose: bool = True,
        convergence_tol: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run EM (approximate) for the MNAR LDS on a single sequence.

        This alternates:
            - E-step: MNAR-aware EKF + RTS smoother
            - M-step: closed-form updates for (mu0, Sigma0, A, Q, C, R)
                      + gradient-ascent updates for phi

        Notes
        -----
        * We use the smoothed posterior (mu_smooth, Sigma_smooth) as
          the approximate q(z_{0:T}).
        * Cross-covariances E[z_t z_{t-1}^T] for A,Q are approximated
          using outer products of smoothed means:
              S_{t,t-1} ≈ mu_t mu_{t-1}^T
          This is not exact RTS EM, but is usually a reasonable
          approximation and keeps the code simple.
        * Phi is updated by logistic regression on smoothed states:
              m_{t,d} ~ Bernoulli( sigma( phi_d^T mu_{t|T} ) )
          using a few gradient steps per EM iteration.

        Parameters
        ----------
        x_t : np.ndarray, shape (T, D)
            Speed panel (NaNs allowed, must match m_t).
        m_t : np.ndarray, shape (T, D)
            Missingness indicators (1 = missing, 0 = observed).
        num_iters : int
            Number of EM iterations.
        update_phi : bool
            If True, update phi each iteration using gradient ascent.
        phi_steps : int
            Number of gradient steps for each detector per EM iter.
        phi_lr : float
            Learning rate for phi updates.
        verbose : bool
            If True, print progress each iteration.
        convergence_tol : Optional[float]
            If not None, perform early stopping when the maximum
            relative change in {A, C, phi} between successive
            iterations drops below this threshold.

        Returns
        -------
        history : dict
            Simple history with parameter snapshots per iteration.
        """
        T, D = x_t.shape
        K = self.params.K

        history = {
            "A": [],
            "Q": [],
            "C": [],
            "R_diag": [],
            "phi": [],
        }

        for it in range(num_iters):
            if verbose:
                print(f"\n=== EM iteration {it + 1}/{num_iters} ===")

            # -------------------------------
            # E-step: EKF + RTS smoother
            # -------------------------------
            ekf_res = self.ekf_forward(x_t=x_t, m_t=m_t)
            smooth_res = self.rts_smoother(ekf_res)

            mu_smooth = smooth_res["mu_smooth"]        # (T, K)
            Sigma_smooth = smooth_res["Sigma_smooth"]  # (T, K, K)

            # S_t = E[z_t z_t^T] = Σ_{t|T} + μ_{t|T} μ_{t|T}^T
            S_t = Sigma_smooth + np.einsum("ti,tj->tij", mu_smooth, mu_smooth)

            # ------------------------------------------------
            # M-step: Initial state (mu0, Sigma0)
            # ------------------------------------------------
            mu0_new = mu_smooth[0].copy()         # (K,)
            Sigma0_new = Sigma_smooth[0].copy()   # (K, K)

            # ------------------------------------------------
            # M-step: Dynamics (A, Q) using approximate stats
            # ------------------------------------------------
            # Sum over t = 1..T-1 (since we use z_{t-1})
            S_t_sum = np.sum(S_t[1:], axis=0)          # sum S_t, t>=1
            S_tminus1_sum = np.sum(S_t[:-1], axis=0)   # sum S_{t-1}, t=0..T-2

            # Approximate cross S_{t,t-1} ~ mu_t mu_{t-1}^T
            S_cross_sum = np.zeros((K, K), dtype=float)
            for t in range(1, T):
                S_cross_sum += np.outer(mu_smooth[t], mu_smooth[t - 1])

            # A_new = (sum_t S_{t,t-1}) (sum_t S_{t-1})^{-1}
            A_new = S_cross_sum @ np.linalg.inv(S_tminus1_sum + 1e-8 * np.eye(K))

            # --- Regularization: shrink A toward identity to keep dynamics stable ---
            lam_A = 0.05  # e.g., 5% pull toward I; tune if needed
            A_new = (1.0 - lam_A) * A_new + lam_A * np.eye(K)

            # Q_new from:
            #   Q = 1/(T-1) sum_t [
            #         S_t
            #       - A S_{t,t-1}^T
            #       - S_{t,t-1} A^T
            #       + A S_{t-1} A^T
            #     ]
            Q_accum = np.zeros((K, K), dtype=float)
            for t in range(1, T):
                S_curr = S_t[t]
                S_prev = S_t[t - 1]

                # Approximate cross-cov S_{t,t-1}
                S_cross = np.outer(mu_smooth[t], mu_smooth[t - 1])

                term = (
                    S_curr
                    - A_new @ S_cross.T
                    - S_cross @ A_new.T
                    + A_new @ S_prev @ A_new.T
                )
                Q_accum += term

            Q_new = Q_accum / (T - 1)
            
            # Symmetrize and add jitter for numerical stability
            Q_new = 0.5 * (Q_new + Q_new.T)
            Q_new += 1e-6 * np.eye(K)

            # --- Regularization: shrink Q toward an isotropic prior and cap its scale ---
            lam_Q = 0.3          # how much to pull Q toward the prior; tune if needed
            Q_prior = 0.1 * np.eye(K)  # prior process noise level
            Q_new = (1.0 - lam_Q) * Q_new + lam_Q * Q_prior

            # Cap the overall scale of Q to avoid exploding dynamics
            max_trace = 30.0    # maximum allowed trace of Q; tune if needed
            tr_Q = float(np.trace(Q_new))
            if tr_Q > max_trace:
                Q_new *= max_trace / tr_Q

            # ------------------------------------------------
            # M-step: Emissions (C, R) with masking
            # ------------------------------------------------
            C_new = np.zeros_like(self.params.C)       # (D, K)
            R_new = np.zeros_like(self.params.R)       # (D, D)

            for d in range(D):
                # Times where detector d is observed
                obs_idx = np.where((m_t[:, d] == 0) & (~np.isnan(x_t[:, d])))[0]
                if obs_idx.size == 0:
                    # No data for this detector; keep old params
                    C_new[d] = self.params.C[d]
                    R_new[d, d] = self.params.R[d, d]
                    continue

                # Vectorized sufficient statistics for C_d and R_dd
                # Shapes:
                #   S_obs  : (#obs, K, K)
                #   mu_obs : (#obs, K)
                #   x_obs  : (#obs,)
                S_obs = S_t[obs_idx]                # (N_obs, K, K)
                mu_obs = mu_smooth[obs_idx]         # (N_obs, K)
                x_obs = x_t[obs_idx, d].astype(float)  # (N_obs,)

                # Denominator: sum_t S_t
                S_sum_d = S_obs.sum(axis=0)         # (K, K)

                # Numerator: sum_t x_{t,d} mu_t^T
                # (N_obs, 1) * (N_obs, K) -> (N_obs, K) -> sum over obs
                num_d = (x_obs[:, None] * mu_obs).sum(axis=0)  # (K,)

                # C_d = num_d (sum_t S_t)^{-1}
                C_d_row = num_d @ np.linalg.inv(S_sum_d + 1e-8 * np.eye(K))
                C_new[d, :] = C_d_row

                # Now compute R_dd
                # For each t:
                #   x_{t,d}^2
                #   - 2 x_{t,d} (C_d mu_t)
                #   + C_d S_t C_d^T
                # Vectorized:
                #   preds  = C_d mu_t  for all obs
                #   C_S_Ct = C_d S_t C_d^T  for all obs

                # preds = C_d mu_t  -> (N_obs,)
                preds = mu_obs @ C_d_row            # (N_obs,)

                # C_S_C = C_d S_t C_d^T per timestep t
                # einsum index explanation:
                #   i   : latent index (for C_d_row left)
                #   t,i,j : S_obs[t, i, j]
                #   j   : latent index (for C_d_row right)
                # Result: (t)
                C_S_C = np.einsum("i,tij,j->t", C_d_row, S_obs, C_d_row)  # (N_obs,)

                err = x_obs**2 - 2.0 * x_obs * preds + C_S_C             # (N_obs,)
                R_dd = float(err.mean())
                R_dd = max(R_dd, 1e-6)    # Ensure non-negative + jitter
                R_new[d, d] = R_dd

            # ------------------------------------------------
            # M-step: Missingness parameters phi (logistic)
            # ------------------------------------------------
            phi_new = self.params.phi.copy()  # (D, K)
            if update_phi:
                # Design matrix: smoothed means z_t ≈ mu_{t|T}
                Z = mu_smooth  # (T, K)

                for d in range(D):
                    # Targets: m_{t,d} in {0,1}
                    y = m_t[:, d].astype(float)  # (T,)

                    # Current phi_d
                    phi_d = phi_new[d].copy()   # (K,)

                    for _ in range(phi_steps):
                        # logits = z_t^T phi_d
                        logits = Z @ phi_d                      # (T,)
                        p = _sigmoid(logits)                    # (T,)

                        # Gradient of log-likelihood:
                        #   sum_t (y_t - p_t) z_t
                        grad = Z.T @ (y - p)                    # (K,)

                        # Gradient-ascent update (scaled by T)
                        phi_d += (phi_lr / T) * grad

                    phi_new[d] = phi_d

            # ------------------------------------------------
            # Compute relative parameter change (for early stop)
            # ------------------------------------------------
            max_rel_change = None
            if convergence_tol is not None and it > 0:
                # Helper to compute relative Frobenius change
                def _rel_change(new: np.ndarray, old: np.ndarray) -> float:
                    num = np.linalg.norm(new - old)
                    denom = np.linalg.norm(old) + 1e-8
                    return num / denom

                prev_params = self.params
                delta_A = _rel_change(A_new, prev_params.A)
                delta_C = _rel_change(C_new, prev_params.C)
                delta_phi = _rel_change(phi_new, prev_params.phi)
                max_rel_change = max(delta_A, delta_C, delta_phi)

            # ------------------------------------------------
            # Update the parameter container
            # ------------------------------------------------
            self.params = MNARParams(
                A=A_new,
                Q=Q_new,
                C=C_new,
                R=R_new,
                mu0=mu0_new,
                Sigma0=Sigma0_new,
                phi=phi_new,
            )

            # Log history (e.g., for debugging)
            history["A"].append(A_new.copy())
            history["Q"].append(Q_new.copy())
            history["C"].append(C_new.copy())
            history["R_diag"].append(np.diag(R_new).copy())
            history["phi"].append(phi_new.copy())

            if verbose:
                mean_R = float(np.mean(np.diag(R_new)))
                print(f"  A norm: {np.linalg.norm(A_new):.3f}")
                print(f"  Q trace: {np.trace(Q_new):.3f}")
                print(f"  mean diag(R): {mean_R:.3f}")
                if max_rel_change is not None:
                    print(f"  max relative param change: {max_rel_change:.3e}")

            # ------------------------------------------------
            # Early stopping condition
            # ------------------------------------------------
            if (
                convergence_tol is not None
                and max_rel_change is not None
                and max_rel_change < convergence_tol
            ):
                if verbose:
                    print(
                        f"  Early stopping at iter {it + 1} "
                        f"(Δ={max_rel_change:.3e} < tol={convergence_tol:.1e})"
                    )
                break

        return history

    # --------------------------------------------------------
    # Calculation of log-likelihood
    # --------------------------------------------------------
    def compute_log_likelihood(
        self,
        x_t: np.ndarray,
        m_t: np.ndarray,
        ekf_results: Dict[str, np.ndarray],
    ) -> float:
        """
        Compute log-likelihood of observed data under the MNAR/MAR LDS.

        This uses the EKF forward pass to compute the one-step-ahead
        predictive log-likelihood at each time step, then sums over t.

        Parameters
        ----------
        x_t : np.ndarray, shape (T, D)
            Speed panel (NaNs allowed, must match m_t).
        m_t : np.ndarray, shape (T, D)
            Missingness indicators (1 = missing, 0 = observed).

        Returns
        -------
        log_likelihood : float
            Total log-likelihood of observed data.
        """
        mu_pred = ekf_results["mu_pred"]          # (T, K)
        Sigma_pred = ekf_results["Sigma_pred"]    # (T, K, K)

        params = self.params
        A, Q, C, R, phi = params.A, params.Q, params.C, params.R, params.phi
        mu_t = params.mu0
        Sigma_t = params.Sigma0

        T, _ = x_t.shape
        log_likelihood = 0.0

        for t in range(T):
            if t > 0:
                mu_t = mu_pred[t]                  # (K,)
                Sigma_t = Sigma_pred[t]            # (K, K)

            # --- Speed block ---
            mask_row = m_t[t]                   # (D,)
            observed_idx = np.where(mask_row == 0)[0]
            if observed_idx.size > 0:
                y_x = x_t[t, observed_idx].astype(float)  # (|O_t|,)
                C_x = C[observed_idx, :]                  # (|O_t|, K)
                R_x = R[np.ix_(observed_idx, observed_idx)]  # (|O_t|, |O_t|)

                h_x = C_x @ mu_t                           # (|O_t|,)
                S_x = C_x @ Sigma_t @ C_x.T + R_x         # (|O_t|, |O_t|)

                diff_x = y_x - h_x
                try:                         # (|O_t|,) Innovation
                    ll_x = -0.5 * (
                        diff_x.T @ np.linalg.inv(S_x) @ diff_x
                        + np.linalg.slogdet(2.0 * np.pi * S_x)[1]
                    )
                except np.linalg.LinAlgError:
                    ll_x = -np.inf
                    
                log_likelihood += ll_x
        return log_likelihood

    # --------------------------------------------------------
    # Reconstruction & forecasting utilities
    # --------------------------------------------------------

    def reconstruct_from_smoother(
        self,
        mu_smooth: np.ndarray,
        Sigma_smooth: np.ndarray,
    ) -> list[np.ndarray, np.ndarray]:
        """
        Reconstruct x_t from smoothed latent states.

        Uses:
            x_hat_t = C * mu_{t|T}

        Parameters
        ----------
        mu_smooth : np.ndarray, shape (T, K)
            Smoothed latent means from rts_smoother().

        Returns
        -------
        x_hat : np.ndarray, shape (T, D)
            Reconstructed speed panel.
        cov : np.ndarray, shape (T, D, D)
            Reconstructed covariance matrices.
        """
        C = self.params.C
        # Matrix multiply for all T at once:
        # (T, K) @ (K, D)^T  => (T, D)
        x_hat = mu_smooth @ C.T

        cov = C @ Sigma_smooth @ C.T + self.params.R

        return x_hat, cov

    def k_step_forecast(
        self,
        mu_filt: np.ndarray,
        Sigma_filt: np.ndarray,
        start_idx: int,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        k-step-ahead forecast starting from time 'start_idx'.

        This corresponds to forecasting from the end of a blackout:
        you condition on data up to time 'start_idx', then propagate
        forward using the dynamics.

        Parameters
        ----------
        mu_filt : np.ndarray, shape (T, K)
            Filtered means from ekf_forward().
        Sigma_filt : np.ndarray, shape (T, K, K)
            Filtered covariances from ekf_forward().
        start_idx : int
            Index 'b' where the blackout ends. We forecast from t = b.
        k : int
            Horizon length (in steps), e.g. 1, 3, or 6.

        Returns
        -------
        mean_x : np.ndarray, shape (D,)
            Forecast mean x_{b+k}.
        cov_x : np.ndarray, shape (D, D)
            Forecast covariance of x_{b+k}.
        """
        A, Q, C, R = self.params.A, self.params.Q, self.params.C, self.params.R

        # Start from filtered posterior at time b
        mu = mu_filt[start_idx].copy()        # (K,)
        Sigma = Sigma_filt[start_idx].copy()  # (K, K)

        # Propagate state forward k steps with dynamics
        for _ in range(k):
            mu = A @ mu
            Sigma = A @ Sigma @ A.T + Q

        # Map to observation space:
        #   x_{b+k} ~ N(C mu, C Σ C^T + R)
        mean_x = C @ mu                          # (D,)
        cov_x = C @ Sigma @ C.T + R              # (D, D)

        return mean_x, cov_x

