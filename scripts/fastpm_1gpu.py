import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax_cosmo as jc

from jax.experimental.ode import odeint

from jaxpm.painting import cic_paint
from jaxpm.pm import linear_field, lpt, make_ode_fn

import matplotlib.pyplot as plt
import numpy as np
import diffrax
from diffrax import diffeqsolve, ODETerm, PIDController, SaveAt
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a simulation')

    parser.add_argument('-s', '--size', type=int, default=64)
    parser.add_argument('-b', '--box_size', type=int, default=200)
    parser.add_argument('-o', '--output', type=str, default='.')
    parser.add_argument('-ode', '--ode', action='store_true')

    args = parser.parse_args()

    mesh_shape = [args.size] * 3
    box_size = [float(args.box_size)] * 3
    snapshots = jnp.linspace(0.02, .9, 10)
    output_dir = f"{args.output}/mesh_{args.size}/box_{args.box_size}"
    use_ode = args.ode
    os.makedirs(output_dir, exist_ok=True)

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
        particles = jnp.stack(
            jnp.meshgrid(*[jnp.arange(s) for s in mesh_shape]),
            axis=-1).reshape([-1, 3])

        cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)

        # Initial displacement
        dx, p, f = lpt(cosmo, initial_conditions, particles, 0.1)
        field = dx + particles

        if use_ode:
            ode_fn = make_ode_fn(mesh_shape)
            term = ODETerm(
                lambda t, state, args: jnp.stack(ode_fn(state, t, args), axis=0))
            solver = diffrax.Dopri8()

            stepsize_controller = diffrax.PIDController(atol=1e-7,rtol=1e-7)
            res = diffeqsolve(term,
                            solver,
                            t0=0.01,
                            t1=1.,
                            dt0=0.1,
                            y0=jnp.stack([particles + dx, p], axis=0),
                            saveat=SaveAt(ts=snapshots),
                            args=cosmo,
                            stepsize_controller=stepsize_controller)

            # res = odeint(make_ode_fn(mesh_shape), [particles + dx, p],
            #              snapshots,
            #              cosmo,
            #              rtol=1e-5,
            #              atol=1e-5)

            return field, res, initial_conditions
        else:
            return field, None, initial_conditions

    particle_field, res, initial_conditions = run_simulation(0.25, 0.8)

    res = res.ys

    # Plotting and saving to PNG
    print(f"Shape of res is {res.shape}")
    # num_snapshots = len(res[0]) if use_ode else 0
    num_snapshots = res.shape[0] if use_ode else 0

    plt.figure(figsize=(12 * (num_snapshots + 1), 6 * 3))
    proj_axis = 0

    # Plot initial conditions
    plt.subplot(1, num_snapshots + 2, 1)
    plt.imshow(initial_conditions.sum(axis=proj_axis), cmap='magma')
    plt.xlabel('Mpc/h')
    plt.ylabel('Mpc/h')
    plt.title('Initial conditions')

    # Plot LPT field
    plt.subplot(1, num_snapshots + 2, 2)
    field = cic_paint(jnp.zeros(mesh_shape), particle_field)
    plt.imshow((field.sum(axis=proj_axis) + 1),
               cmap='magma',
               extent=[0, box_size[0], 0, box_size[1]])
    plt.xlabel('Mpc/h')
    plt.ylabel('Mpc/h')
    plt.title(f'LPT density field at z=0')

    np.save(f'{output_dir}/field_{args.size}_{args.box_size}_single.npy',
            field)
    np.save(
        f'{output_dir}/particle_field_{args.size}_{args.box_size}_single.npy',
        particle_field)
    np.save(f'{output_dir}/init_{args.size}_{args.box_size}_single.npy',
            initial_conditions)
    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in mesh_shape]),
                          axis=-1).reshape([-1, 3])

    if use_ode:
        for i, step in enumerate(res):

            solut = cic_paint(jnp.zeros(mesh_shape), step[0])
            np.save(
                f'{output_dir}/solution_{i}_{args.size}_{args.box_size}_single.npy',
                solut)
            np.save(
                f'{output_dir}/particle_solution_{i}_{args.size}_{args.box_size}_single.npy',
                step)

            print(f"Shape of field is {solut.shape}")
            plt.subplot(1, num_snapshots + 2, 3 + i)
            plt.imshow((solut.sum(axis=proj_axis) + 1),
                       cmap='magma',
                       extent=[0, box_size[0], 0, box_size[1]])
            plt.xlabel('Mpc/h')
            plt.ylabel('Mpc/h')
            plt.title(f'ODT at {i}')

    plt.savefig(f'{output_dir}/single_dev_simulation_results.png')
