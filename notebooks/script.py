import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)

from jax.experimental.ode import odeint

from jaxpm.painting import cic_paint
from jaxpm.pm import linear_field, lpt, make_ode_fn

shape = 64
mesh_shape = [shape, shape, shape]
box_size = [shape, shape, shape]
snapshots = jnp.linspace(0.1, 1., 2)


# @jax.jit
def run_simulation(omega_c, sigma8):
    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(
        jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk
                                                  ).reshape(x.shape)

    # Create initial conditions1
    initial_conditions = linear_field(mesh_shape,
                                      box_size,
                                      pk_fn,
                                      seed=jax.random.PRNGKey(0))
    print(f"Initial conditions shape is {initial_conditions.shape}")
    print(f"Initial conditions are {initial_conditions}")
    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in mesh_shape]),
                          axis=-1).reshape([-1, 3])

    print(f"Particles shape is {particles.shape}")
    print(f"Particles are {particles}")
    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
    print(f"cosmo is {cosmo}")
    # Initial displacement
    dx, p, f = lpt(cosmo, initial_conditions, particles, 0.1)

    # Evolve the simulation forward
    res = odeint(make_ode_fn(mesh_shape), [particles + dx, p],
                 snapshots,
                 cosmo,
                 rtol=1e-5,
                 atol=1e-5)

    # Return the simulation volume at requested
    return res[0]


res = run_simulation(0.25, 0.8)
res = run_simulation(0.25, 0.8)

figure = plt.figure(figsize=[10, 5])
subplot = figure.add_subplot(121)
subplot.imshow(cic_paint(jnp.zeros(mesh_shape), res[0]).sum(axis=0),
               cmap='gist_stern')
subplot = figure.add_subplot(122)
subplot.imshow(cic_paint(jnp.zeros(mesh_shape), res[1]).sum(axis=0),
               cmap='gist_stern')

figure.savefig('plot.png')
