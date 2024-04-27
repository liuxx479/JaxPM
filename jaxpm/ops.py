# Module for custom ops, typically mpi4jax
import jax
import jax.numpy as jnp
import jaxdecomp
from jaxdecomp import halo_exchange, slice_pad, slice_unpad
from dataclasses import dataclass
from typing import Tuple
from functools import partial, reduce
from jax import jit
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P,NamedSharding
from jax.experimental.shard_map import shard_map

@dataclass
class ShardingInfo:
    """Class for keeping track of the distribution strategy"""
    global_shape: Tuple[int, int, int] 
    pdims: Tuple[int, int]
    halo_extents: Tuple[int, int, int]
    rank: int = 0
    mesh : Mesh = None

    def __post_init__(self):
        devices = mesh_utils.create_device_mesh(self.pdims[::-1])
        self.mesh = Mesh(devices, axis_names=('z', 'y'))

    # Hash function for the class so it can be used as static argnum
    def __hash__(self) -> int:
        return hash((self.global_shape, self.pdims, self.halo_extents, self.rank))


def fft3d(arr, sharding_info=None):
    """ Computes forward FFT, note that the output is transposed
    """
    if sharding_info is None:
        arr = jnp.fft.fftn(arr).transpose([1, 2, 0])
    else:
        arr = jaxdecomp.pfft3d(arr)
    return arr

def ifft3d(arr, sharding_info=None):
    if sharding_info is None:
        arr = jnp.fft.ifftn(arr.transpose([2, 0, 1]))
    else:
        arr = jaxdecomp.pifft3d(arr)
    return arr



def halo_reduce(arr, sharding_info):
    
    if sharding_info is None:
        return arr

    x_halo = sharding_info.halo_extents[0]
    y_halo = sharding_info.halo_extents[1]
    unpadding = ((x_halo, x_halo), (y_halo, y_halo), (0, 0))
    with sharding_info.mesh:
        arr = halo_exchange(arr,
                            halo_extents=(x_halo, y_halo, 0),
                            halo_periods=(True,True,True),
                            reduce_halo = True)

        arr = slice_unpad(arr, unpadding , sharding_info.pdims)

    
    return arr


def meshgrid3d(shape, sharding_info=None):
    if sharding_info is not None:
        coords = [jnp.arange(sharding_info.global_shape[0]//sharding_info.pdims[1]),
                  jnp.arange(sharding_info.global_shape[1]//sharding_info.pdims[0]), jnp.arange(sharding_info.global_shape[2])]
    else:
        coords = [jnp.arange(s) for s in shape[2:]]

    return jnp.stack(jnp.meshgrid(*coords), axis=-1).reshape([-1, 3])

def zeros(shape, sharding_info=None):
    """ Initialize an array of given global shape
    partitionned if need be accross dimensions.
    """
    if sharding_info is None:
        return jnp.zeros(shape)

    zeros_slice = jnp.zeros([sharding_info.global_shape[0]//sharding_info.pdims[1], \
        sharding_info.global_shape[1]//sharding_info.pdims[0]]+list(sharding_info.global_shape[2:]))

    gspmd_zeros = multihost_utils.host_local_array_to_global_array(zeros_slice ,sharding_info.mesh, P('z' , 'y'))
    return gspmd_zeros


def normal(key, shape, sharding_info=None):
    """ Generates a normal variable for the given
    global shape.
    """
    if sharding_info is None:
        return jax.random.normal(key, shape)

    return jax.random.normal(key,
                            [sharding_info.global_shape[0]//sharding_info.pdims[1], sharding_info.global_shape[1]//sharding_info.pdims[0], sharding_info.global_shape[2]])
