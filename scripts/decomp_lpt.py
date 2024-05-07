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

pdims = (2 , 2)
mesh_shape = [256, 256, 256]
box_size = [200, 200, 200]  # Mpc/h

# Create computing mesgh
devices = mesh_utils.create_device_mesh(pdims)
mesh = Mesh(devices, axis_names=('y', 'z'))
sharding = jax.sharding.NamedSharding(mesh, P('z', 'y'))

### Create all initial distributed tensors ###
local_mesh_shape = [mesh_shape[0]//pdims[1], mesh_shape[1]//pdims[0], mesh_shape[2]]

# Create gaussian field distributed across the mesh
z = jax.make_array_from_single_device_arrays(shape=mesh_shape,
                                             sharding=sharding,
                                             arrays=[jax.random.normal(key, local_mesh_shape)])

# I'm not sure I understand the order of the data I have to provide here
# Order is ZXY column major so YXZ
pos = jax.make_array_from_callback(shape=tuple(mesh_shape+[3]),
                                   sharding=sharding,
                                   data_callback=lambda x: jnp.stack(jnp.meshgrid(jnp.arange(mesh_shape[1])[x[1]],
                                                                                  jnp.arange(mesh_shape[2])[x[0]], 
                                                                                  jnp.arange(mesh_shape[0])), axis=-1))

# this is a cube so mesh shape doesn't matter
# but I think it should be 0 1 2 and 2 for X
kd = np.fft.fftfreq(mesh_shape[0]) * 2 * np.pi # must be mesh_shape[2]
kvec = [jax.make_array_from_callback((mesh_shape[0], 1, 1), 
                                    sharding=jax.sharding.NamedSharding(mesh, P('z')),
                                    data_callback=lambda x: kd.reshape([-1,1,1])[x]),
        jax.make_array_from_callback((1, mesh_shape[1], 1), 
                                    sharding=jax.sharding.NamedSharding(mesh, P(None, 'y')),
                                    data_callback=lambda x: kd.reshape([1,-1,1])[x]),
        kd.reshape([1,1,-1])]

# Checking the size of all local tensors
print('rank =', rank, 'z.local_shape =', 
      z.addressable_data(0).shape,
      'pos.local_shape =', pos.addressable_data(0).shape,
      'kvec[0].local_shape =', kvec[0].addressable_data(0).shape,
      'kvec[1].local_shape =', kvec[1].addressable_data(0).shape)
#############################################

@jax.jit
def forward_fn(z, kvec, pos, a):
    kfield = jaxdecomp.fft.pfft3d(z.astype(jnp.complex64))

    # Rescaling k to physical units
    kvec = [k / box_size[i] * mesh_shape[i]
            for i, k in enumerate(kvec)]
    kz, ky, kx = kvec
    kk = jnp.sqrt(kx**2+ky**2+kz**2)
    
    k = jnp.logspace(-4, 2, 256) # I don't understand why 256?
    pk = jc.power.linear_matter_power(jc.Planck15(), k)
    pk = pk * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]
                ) / (box_size[0] * box_size[1] * box_size[2])

    # Multipliyng the field by the proper power spectrum
    # Interpolate has a vmap in it ... so it is all-gathered since jax does not know how to deal with it
    # it should probably be redone (not just shard-map because it still all-gather)
    delta_k = kfield * jc.scipy.interpolate.interp(kk.flatten(), k, jnp.sqrt(pk)).reshape(kfield.shape)

    # Inverse fourier transform to generate the initial conditions
    initial_conditions = jaxdecomp.fft.pifft3d(delta_k).real
    # Now the result is an X pencil so ZYX for us

    ###  Compute LPT displacement
    kernel = jnp.where(kk == 0, 1., 1./kk) # Laplace kernel
    # Forces have to be a Z pencil because they are going to be IFFT back to X pencil
    forces_k = jnp.stack([delta_k * kernel * 1j * 1 / 6.0 * (8 * jnp.sin(ky) - jnp.sin(2 * ky)),
                          delta_k * kernel * 1j * 1 / 6.0 * (8 * jnp.sin(kx) - jnp.sin(2 * kx)),
                          delta_k * kernel * 1j * 1 / 6.0 * (8 * jnp.sin(kz) - jnp.sin(2 * kz))], 
                          axis=-1)
    
    # Interpolate forces at the position of particles
    pos = pos.reshape([-1,3])
    # I don't understand this very well .. three IFFTs is ok I understand (for each direction) but each on a k ? each k is 3D object?
    initial_force = jnp.stack([cic_read(jaxdecomp.fft.pifft3d(forces_k[..., i]).real, pos)
                               for i in range(3)], axis=-1)
    cosmo = jc.Planck15()
    a = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a) * initial_force

    # Painting resulting particles
    field = cic_paint(jax.numpy.zeros_like(z), pos+dx)

    return initial_conditions, field

with mesh:
    initial_conds, field= forward_fn(z, kvec, pos, a=1.)
    field.block_until_ready()
    
    # Measuring how long it takes to run after the first call
    start = time.time()
    initial_conds, field = forward_fn(z, kvec, pos, a=1.)
    field.block_until_ready()
    end = time.time()

print('success in', end-start, 'seconds')

# Saving results
np.save(f'./field_{rank}.npy', field.addressable_data(0))
np.save(f'./initial_conditions_{rank}.npy', initial_conds.addressable_data(0))