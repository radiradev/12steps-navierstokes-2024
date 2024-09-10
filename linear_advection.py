import marimo

__generated_with = "0.8.13"
app = marimo.App()


@app.cell
def __(mo):
    mo.md("""# 1-D Linear Convection""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        The following is a refactor of the wonderful [CFD Python: 12 Steps to Navier-Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) course by Lorena Barba.

        In this version I am using a `numpy` first approach as this results in more easily readable code. I am also utilizing the interactivity provided by `marimo` to aid to the exploration of the behaviour of the differential equations and integration schemes.


        For a more in depth discussion of the advection equation I highly recommend the video from the MIT course [Introduction to Computational Thinking](https://www.youtube.com/watch?v=Xb-iUwXI78A&t=2063s).
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        The 1-D Linear Convection equation is the simplest, most basic model that can be used to learn something about CFD. It is surprising that this little equation can teach us so much! Here it is:

        $$\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0$$

        With given initial conditions (understood as a *wave*), the equation represents the propagation of that initial *wave* with speed $c$, without change of shape. Let the initial condition be $u(x,0)=u_0(x)$. Then the exact solution of the equation is $u(x,t)=u_0(x-ct)$.

        We discretize this equation in both space and time, using the Forward Difference scheme for the time derivative and the Backward Difference scheme for the space derivative. Consider discretizing the spatial coordinate $x$ into points that we index from $i=0$ to $N$, and stepping in discrete time intervals of size $\Delta t$.

        From the definition of a derivative (and simply removing the limit), we know that:

        $$\frac{\partial u}{\partial x}\approx \frac{u(x+\Delta x)-u(x)}{\Delta x}$$

        Our discrete equation, then, is:

        $$\frac{u_i^{n+1}-u_i^n}{\Delta t} + c \frac{u_i^n - u_{i-1}^n}{\Delta x} = 0 $$

        Where $n$ and $n+1$ are two consecutive steps in time, while $i-1$ and $i$ are two neighboring points of the discretized $x$ coordinate. If there are given initial conditions, then the only unknown in this discretization is $u_i^{n+1}$.  We can solve for our unknown to get an equation that allows us to advance in time, as follows:

        \begin{equation}u_i^{n+1} = u_i^n - c \frac{\Delta t}{\Delta x}(u_i^n-u_{i-1}^n) \end{equation}

        We would focus on a method of stepping through time using a finite difference scheme through Equation (1) as our first endeavour into CFD!
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        One of the first issues we might come across is what do we do when $i=0$? One way of dealing with this would be to implement periodic boundary conditions, meaning $u_{N+1}=u_{0}$ and $u_{-1}=u_{N}$. Since we would like to perform a vectorized operation we can implement this using `np.roll`. 

        Finally we just step through Equation (1) a number of time steps for a given intial condition and save it's evolution.
        """
    )
    return


@app.cell
def __():
    import numpy as np                 
    import matplotlib.pyplot as plt
    import marimo as mo
    return mo, np, plt


@app.cell
def __(np):
    class LinearConvection:
        def __init__(self, x, initial_condition, c=1):
            super().__init__()
            self.x = x
            self.initial_condition = initial_condition

            endpoint = x[-1]
            self.num_grid_points = len(x)
            self.delta_x = endpoint / (self.num_grid_points - 1)
            self.c = c

        def integrate(self, num_time_steps=300, delta_t=0.025):
            trajectory = np.empty((num_time_steps, self.num_grid_points))
            self.u = self.initial_condition.copy()

            cfl = self.c * delta_t / self.delta_x # > 1 results in instability

            for n in range(num_time_steps):
                u_left = np.roll(self.u, 1)

                self.u = self.u - cfl * (self.u - u_left)

                trajectory[n] = self.u

            return trajectory
    return LinearConvection,


@app.cell
def __(LinearConvection, np, num_grid_points):
    num_time_steps = 300
    x = np.linspace(-np.pi, np.pi, num_grid_points.value)
    initial_condition = np.sin(x)

    convection = LinearConvection(x, initial_condition)
    trajectory = convection.integrate(num_time_steps)
    return convection, initial_condition, num_time_steps, trajectory, x


@app.cell
def __(mo):
    num_grid_points = mo.ui.slider(10, 500, label='number of grid points')
    timestep = mo.ui.slider(0, 300 - 1, label='timestep')

    [timestep, num_grid_points]
    return num_grid_points, timestep


@app.cell
def __(plot_trajectory, timestep):
    plot_trajectory(timestep.value)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        Using `marimo` we create interactive variables for the timestep, which allows us to see a particular frame of the simulated trajectory. We also have a variable for the number of grid points, which controls the resolution of the simulation. We can see that going to higher number of grid points results in a smoother wave, however at around 120 we experience instability. 

        This a [well studied phenomenon](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition) and happens when `cfl` in `integrate()` is higher than 1.
        """
    )
    return


@app.cell
def __(convection, plt, trajectory):
    def plot_trajectory(timestep):
        plt.plot(convection.x, trajectory[timestep])
        plt.title(f"$t={timestep}$")
        plt.xlabel("$x$")
        plt.ylabel("$u$")
        return plt.gca()
    return plot_trajectory,


if __name__ == "__main__":
    app.run()
