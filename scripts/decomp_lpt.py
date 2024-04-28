# Execute with `mpirun -n 4 python decomp_lpt.py`
# Works under jaxdecomp v0.0.1rc2
import jax

import jax.numpy as jnp
import jax.lax as lax

import numpy as np
import jaxdecomp
import jax_cosmo as jc
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P
from jaxpm.growth import growth_factor, growth_rate, dGfa
import time

def cic_paint(mesh, positions):
    """ Paints positions onto mesh
    mesh: [nx, ny, nz]
    positions: [npart, 3]
    """
    positions = jnp.expand_dims(positions, 1)
    floor = jnp.floor(positions)
    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0],
                             [0., 0, 1], [1., 1, 0], [1., 0, 1],
                             [0., 1, 1], [1., 1, 1]]])

    neighboor_coords = floor + connection
    kernel = 1. - jnp.abs(positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    neighboor_coords = jnp.mod(neighboor_coords.reshape(
        [-1, 8, 3]).astype('int32'), jnp.array(mesh.shape))

    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1, 2),
        scatter_dims_to_operand_dims=(0, 1, 2))
    mesh = lax.scatter_add(mesh,
                           neighboor_coords,
                           kernel.reshape([-1, 8]),
                           dnums)
    return mesh

def cic_read(mesh, positions):
    """ Paints positions onto mesh
    mesh: [nx, ny, nz]
    positions: [npart, 3]
    """
    positions = jnp.expand_dims(positions, 1)
    floor = jnp.floor(positions)
    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0],
                             [0., 0, 1], [1., 1, 0], [1., 0, 1],
                             [0., 1, 1], [1., 1, 1]]])

    neighboor_coords = floor + connection
    kernel = 1. - jnp.abs(positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    neighboor_coords = jnp.mod(
        neighboor_coords.astype('int32'), jnp.array(mesh.shape))

    return (mesh[neighboor_coords[..., 0],
                 neighboor_coords[..., 1],
                 neighboor_coords[..., 3]]*kernel).sum(axis=-1)


# Initializing distributed JAX operations
jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()

# Setup random keys
master_key = jax.random.PRNGKey(42)
key = jax.random.split(master_key, size)[rank]

pdims = (4, 1)
mesh_shape = [512, 512, 512]
box_size = [200, 200, 200]  # Mpc/h

# Create computing mesgh
devices = mesh_utils.create_device_mesh(pdims[::-1])
mesh = Mesh(devices, axis_names=('x', 'y'))
sharding = jax.sharding.NamedSharding(mesh, P('y', 'x'))

### Create all initial distributed tensors ###
local_mesh_shape = [mesh_shape[0]//pdims[0], mesh_shape[1]//pdims[1], mesh_shape[2]]

# Create gaussian field distributed across the mesh
z = jax.make_array_from_single_device_arrays(shape=mesh_shape,
                                             sharding=sharding,
                                             arrays=[jax.random.normal(key, local_mesh_shape)])

# I'm not sure I understand the order of the data I have to provide here
pos = jax.make_array_from_callback(shape=tuple(mesh_shape+[3]),
                                   sharding=sharding,
                                   data_callback=lambda x: jnp.stack(jnp.meshgrid(jnp.arange(mesh_shape[1])[x[1]],
                                                                                  jnp.arange(mesh_shape[0])[x[0]], 
                                                                                  jnp.arange(mesh_shape[2])), axis=-1))

kd = np.fft.fftfreq(mesh_shape[0]) * 2 * np.pi
kvec = [jax.make_array_from_callback((mesh_shape[0], 1, 1), 
                                    sharding=jax.sharding.NamedSharding(mesh, P('y')),
                                    data_callback=lambda x: kd.reshape([-1,1,1])[x]),
        jax.make_array_from_callback((1, mesh_shape[1], 1), 
                                    sharding=jax.sharding.NamedSharding(mesh, P(None, 'x')),
                                    data_callback=lambda x: kd.reshape([1,-1,1])[x]),
        kd.reshape([1,1,-1])]

# Checking the size of all local tensors
print('rank =', rank, 'z.local_shape =', 
      z.addressable_shards[0].data.shape,
      'pos.local_shape =', pos.addressable_shards[0].data.shape,
      'kvec[0].local_shape =', kvec[0].addressable_shards[0].data.shape,
      'kvec[1].local_shape =', kvec[1].addressable_shards[0].data.shape)
#############################################

@jax.jit
def forward_fn(z, kvec, pos, a):
    kfield = jaxdecomp.fft.pfft3d(z.astype(jnp.complex64))

    # Rescaling k to physical units
    kvec = [k / box_size[i] * mesh_shape[i]
            for i, k in enumerate(kvec)]
    kx, ky, kz = kvec
    kk = jnp.sqrt(kx**2+ky**2+kz**2)
    
    k = jnp.logspace(-4, 2, 256)
    pk = jc.power.linear_matter_power(jc.Planck15(), k)
    pk = pk * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]
                ) / (box_size[0] * box_size[1] * box_size[2])

    # Multipliyng the field by the proper power spectrum
    delta_k = kfield * jc.scipy.interpolate.interp(kk.flatten(), k, jnp.sqrt(pk)).reshape(kfield.shape)

    # Inverse fourier transform to generate the initial conditions
    initial_conditions = jaxdecomp.fft.pifft3d(delta_k).real

    ###  Compute LPT displacement
    kernel = jnp.where(kk == 0, 1., 1./kk) # Laplace kernel
    forces_k = jnp.stack([delta_k * kernel * 1j * 1 / 6.0 * (8 * jnp.sin(kz) - jnp.sin(2 * kz)),
                          delta_k * kernel * 1j * 1 / 6.0 * (8 * jnp.sin(kx) - jnp.sin(2 * kx)),
                          delta_k * kernel * 1j * 1 / 6.0 * (8 * jnp.sin(ky) - jnp.sin(2 * ky))], 
                          axis=-1)
    
    # Interpolate forces at the position of particles
    pos = pos.reshape([-1,3])
    initial_force = jnp.stack([cic_read(jaxdecomp.fft.pifft3d(forces_k[..., i]).real, pos)
                               for i in range(3)], axis=-1)
    cosmo = jc.Planck15()
    a = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a) * initial_force
    p = a**2 * growth_rate(cosmo, a) * jnp.sqrt(jc.background.Esqr(cosmo, a)) * dx
    f = a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a)) * dGfa(cosmo, a) * initial_force

    # Painting resulting particles
    field = cic_paint(jax.numpy.zeros_like(z), pos+dx)

    return initial_conditions, field

with mesh:
    initial_conds, field= forward_fn(z, kvec, pos, a=1.)
    # Measuring how long it takes to run
    start = time.time()
    initial_conds, field = forward_fn(z, kvec, pos, a=1.)
    field.block_until_ready()
    end = time.time()

print('success in', end-start, 'seconds')

# Retrieve the results
gathered_initial_conditions = multihost_utils.process_allgather(initial_conds, tiled=True)
gathered_field = multihost_utils.process_allgather(field, tiled=True)

if rank == 0:
    np.save('./field.npy', gathered_field)
    np.save('./initial_conditions.npy', gathered_initial_conditions)
