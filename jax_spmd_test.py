import jax

jax.distributed.initialize()

print(jax.devices())