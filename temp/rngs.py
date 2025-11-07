import jax


key = jax.random.key(10)
print(f"DEBUGPRINT[80]: rngs.py:4: key={key}")
another = jax.random.split(key, 2)
a, b = jax.random.split(key, 2) 
print(f"DEBUGPRINT[78]: rngs.py:5: b={b.shape}")
print(f"DEBUGPRINT[77]: rngs.py:5: a={a}")
print(f"DEBUGPRINT[79]: rngs.py:5: another={another}")
