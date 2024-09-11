import marimo

__generated_with = "0.8.13"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    return Axes3D, mo, np, plt


@app.cell(hide_code=True)
def __(mo):
    mo.md("""# 2D Diffusion""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        For a derivation of the diffusion equation I recommend [this](https://www.youtube.com/watch?v=a3V0BJLIo_c) wonderful video by Grant Sanderson. Here is the differential equation we would be looking at: 

        $$\frac{\partial u}{\partial t} = \nu \frac{\partial ^2 u}{\partial x^2} + \nu \frac{\partial ^2 u}{\partial y^2}$$

        We are going to use a second order discretization scheme, with our forward difference in time and two second-order derivatives. 
        $$\frac{u_{i,j}^{n+1} - u_{i,j}^n}{\Delta t} = \nu \frac{u_{i+1,j}^n - 2 u_{i,j}^n + u_{i-1,j}^n}{\Delta x^2} + \nu \frac{u_{i,j+1}^n-2 u_{i,j}^n + u_{i,j-1}^n}{\Delta y^2}$$

        We reorganize the discretized equation and solve for $u_{i,j}^{n+1}$

        $$
        \begin{split}
        u_{i,j}^{n+1} = u_{i,j}^n &+ \frac{\nu \Delta t}{\Delta x^2}(u_{i+1,j}^n - 2 u_{i,j}^n + u_{i-1,j}^n) \\
        &+ \frac{\nu \Delta t}{\Delta y^2}(u_{i,j+1}^n-2 u_{i,j}^n + u_{i,j-1}^n)
        \end{split}$$
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""We can again use an array based approach, which would be faster and more readable instead of using nested loops. We can see in he equation above that to update the current cell $u_{i, j}$ we need to look at second order difference in the $y$ and $x$ directions (the laplacian). We can again implement this using `np.roll`.""")
    return


@app.cell
def __(np):
    class TwoDimDiffusion:
        def __init__(self, xy, initial_condition, diffusivity=5e-2):
            self.xy = xy
            self.initial_condition = initial_condition
            self.num_grid_points = len(xy)

            endpoint = xy[-1]
            self.delta = endpoint / (self.num_grid_points - 1)
            self.diffusivity = diffusivity

        def integrate(self, num_time_steps=300, delta_t=2.5e-3):
            trajectory = np.empty((num_time_steps, *self.initial_condition.shape))
            u = self.initial_condition.copy()

            const_factor = self.diffusivity * delta_t / self.delta**2

            for n in range(num_time_steps):
                laplacian = self._compute_laplacian(u)
                u += const_factor * laplacian
                trajectory[n] = u

            return trajectory

        def _compute_laplacian(self, u):
            return (
                np.roll(u, 1, axis=0)  # left
                + np.roll(u, -1, axis=0)  # right
                + np.roll(u, 1, axis=1)  # up
                + np.roll(u, -1, axis=1)  # down
                - 4 * u  # center
            )
    return TwoDimDiffusion,


@app.cell(hide_code=True)
def __(mo):
    mo.md("""Play around with different initial conditions and see how the diffusion process evolves.""")
    return


@app.cell
def __(mo):
    condition_types = ["Sine Wave", "Square Wave", "Gaussian"]
    condition_select = mo.ui.dropdown(
        options=condition_types,
        value="Sine Wave",
        label="Select Initial Condition",
    )
    condition_select
    return condition_select, condition_types


@app.cell
def __(TwoDimDiffusion, condition_select, create_initial_condition):
    xy, u0 = create_initial_condition(30, condition_select.value)
    diffusion = TwoDimDiffusion(xy=xy, initial_condition=u0)
    trajectory = diffusion.integrate()
    return diffusion, trajectory, u0, xy


@app.cell
def __(mo):
    timestepper = mo.ui.slider(start=0, stop=300 - 1, label="timestep")
    azimuth = mo.ui.slider(start=15, stop=360, label="viewing angle")
    [timestepper, azimuth]
    return azimuth, timestepper


@app.cell
def __(azimuth, plot, timestepper):
    plot(timestepper.value, azimuth.value)
    return


@app.cell
def __(np):
    def create_initial_condition(num_grid_points, condition_type):
        xy = np.linspace(0, 1, num_grid_points)
        X, Y = np.meshgrid(xy, xy)

        if condition_type == "Sine Wave":
            u0 = 0.5 * (np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) + 1)

        elif condition_type == "Square Wave":
            u0 = np.zeros((num_grid_points, num_grid_points))
            mask = (0.25 < X) & (X < 0.75) & (0.25 < Y) & (Y < 0.75)
            u0[mask] = 1.0

        elif condition_type == "Gaussian":
            sigma = 0.1
            u0 = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * sigma**2))
        else:
            raise ValueError(f"Unknown condition type: {condition_type}")

        return xy, u0
    return create_initial_condition,


@app.cell
def __(np, plt, trajectory, xy):
    def plot(timestep, azimuth):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        X, Y = np.meshgrid(xy, xy)

        # Get the maximum temperature (concentration) in the entire trajectory
        max_temp = np.max(trajectory)

        surf = ax.plot_surface(
            X,
            Y,
            trajectory[timestep],
            cmap="coolwarm",
            edgecolor="none",
            vmin=0,
            vmax=max_temp,
        )

        ax.set_title(f"2D Diffusion Process\nTime step: {timestep}", fontsize=13)
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_zlabel("Concentration", fontsize=12)
        ax.view_init(elev=30, azim=azimuth)  # Set viewing angle

        # Set consistent z-axis limits
        ax.set_zlim(0, max_temp)

        # Add colorbar with the same limits as the z-axis
        fig.colorbar(
            surf, ax=ax, shrink=0.5, aspect=10, label="Concentration", pad=0.1
        )

        plt.tight_layout()
        return fig
    return plot,


if __name__ == "__main__":
    app.run()
