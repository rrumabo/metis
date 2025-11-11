import numpy as np


def heat_step(T, kappa, dx, dy, dt):
    """One explicit Euler step of the 2D heat equation with periodic BCs.

    ∂T/∂t = κ (T_xx + T_yy)

    T: (ny, nx) array
    kappa: thermal diffusivity
    """
    ny, nx = T.shape
    T_ext = np.pad(T, pad_width=1, mode="wrap")

    d2x = (T_ext[1:-1, 2:] - 2.0 * T_ext[1:-1, 1:-1] + T_ext[1:-1, :-2]) / dx ** 2
    d2y = (T_ext[2:, 1:-1] - 2.0 * T_ext[1:-1, 1:-1] + T_ext[:-2, 1:-1]) / dy ** 2

    return T + dt * kappa * (d2x + d2y)


def run_heat_demo(nx, ny, dx, dy, dt, t_end, base_temp=35.0, kappa=1e-5):
    """Run a toy 2D heat diffusion simulation.

    Starts from a hot blob in the center and diffuses it out.
    """
    # initial field: base + Gaussian bump in center
    y = np.linspace(-1.0, 1.0, ny)
    x = np.linspace(-1.0, 1.0, nx)
    X, Y = np.meshgrid(x, y)
    bump = np.exp(-(X**2 + Y**2) / 0.1)
    T = base_temp + 5.0 * bump  # 5°C anomaly

    n_steps = int(t_end / dt)
    for _ in range(n_steps):
        T = heat_step(T, kappa=kappa, dx=dx, dy=dy, dt=dt)

    return T