import jax
import jax.numpy as jnp
import jax_cosmo as jc

from jax.experimental.ode import odeint

from jaxpm.painting import cic_paint
from jaxpm.pm import linear_field, lpt, make_ode_fn

import matplotlib.pyplot as plt

mesh_shape = [64, 64, 64]
box_size = [64., 64., 64.]
snapshots = jnp.linspace(0.1, 1., 4)


@jax.jit
def run_simulation(omega_c, sigma8):
    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(
        jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk
                                                  ).reshape(x.shape)

    # Create initial conditions
    initial_conditions = linear_field(mesh_shape,
                                      box_size,
                                      pk_fn,
                                      seed=jax.random.PRNGKey(0))

    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in mesh_shape]),
                          axis=-1).reshape([-1, 3])

    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)

    # Initial displacement
    dx, p, f = lpt(cosmo, initial_conditions, particles, 0.1)
    field = dx + particles

    # Evolve the simulation forward
    res = odeint(make_ode_fn(mesh_shape), [particles + dx, p],
                 snapshots,
                 cosmo,
                 rtol=1e-5,
                 atol=1e-5)

    # Return the simulation volume at requested
    return field, res, initial_conditions


field, res, initial_conditions = run_simulation(0.25, 0.8)

# Plotting and saving to PNG
num_snapshots = len(snapshots)
plt.figure(figsize=(12 * (num_snapshots + 1), 6 * 3))
proj_axis = 0

# Plot initial conditions
plt.subplot(1, num_snapshots + 1, 1)
plt.imshow(initial_conditions.sum(axis=proj_axis), cmap='magma')
plt.xlabel('Mpc/h')
plt.ylabel('Mpc/h')
plt.title('Initial conditions')

# Plot LPT field
plt.subplot(1, num_snapshots + 1, 2)
field = cic_paint(jnp.zeros(mesh_shape), field)
plt.imshow(jnp.log(field.sum(axis=proj_axis) + 1),
           cmap='magma',
           extent=[0, box_size[0], 0, box_size[1]])

print(f"Snapshots are {snapshots}")
print(f"Number of res is {len(res)}")
# Plot each snapshot
for i, snapshot in enumerate(snapshots):

    field = cic_paint(jnp.zeros(mesh_shape), res[0][i])
    print(f"Shape of field is {field.shape}")
    plt.subplot(1, num_snapshots + 1, i + 2)
    plt.imshow(jnp.log10(field.sum(axis=proj_axis) + 1),
               cmap='magma',
               extent=[0, box_size[0], 0, box_size[1]])
    plt.xlabel('Mpc/h')
    plt.ylabel('Mpc/h')
    plt.title(f'LPT density field at z={1/snapshot - 1:.2f}')

plt.savefig('single_dev_simulation_results.png')
