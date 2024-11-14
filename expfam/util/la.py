import jax
import jax.numpy as jnp
import math
import numpy as np
from jax import vmap

# inv(L*L.T)*Y
def invcholp(L, Y):
    D = jax.scipy.linalg.solve_triangular(L, Y, lower=True)
    B = jax.scipy.linalg.solve_triangular(L.T, D, lower=False)
    return B

# inv(X)*Y
invmp = lambda X, Y: invcholp(jax.linalg.cholesky(X), Y)

# batched outer product
outer = lambda x, y: x[...,None]*y[...,None,:]

# batched transpose
transpose = lambda _: jnp.swapaxes(_, -1, -2)

# batched matrix vector / vector (transpose) matrix product
mvp = lambda X, v: jnp.matmul(X, v[...,None]).squeeze(-1)
vmp = lambda v, X: jnp.matmul(v[...,None,:], X).squeeze(-2)
mmp = jnp.matmul

# batched vector dot product
vdot = lambda x, y: jnp.sum(x*y, -1)

# batched symmetrize
symmetrize = lambda _: .5*(_ + transpose(_))

def submatrix(x, rowmask, colmask):
    return x[outer(rowmask,colmask)].reshape(np.sum(rowmask), np.sum(colmask))

def isposdefh(h):
    return jax.numpy.linalg.eigh(h)[0][...,0] > 0

def diagm(x):
    *shape_prefix, D = x.shape
    nbatches = np.prod(shape_prefix, dtype=np.int64)
    X = vmap(jnp.diag)(x.reshape(nbatches,D))
    return X.reshape(*shape_prefix, D, D)

def diagv(X):
    *shape_prefix, D = X.shape[:-1]
    nbatches = np.prod(shape_prefix, dtype=np.int64)
    V = vmap(jnp.diag)(X.reshape(nbatches,D,D))
    return V.reshape(*shape_prefix, D)

def trilm(x):
    shape_prefix = x.shape[:-1]
    nbatches = np.prod(shape_prefix, dtype=np.int64)
    D = (-1 + math.isqrt(1 + 8*x.shape[-1]))//2
    X = vmap(lambda _: jnp.zeros((D,D)).at[jnp.tril_indices(D)].set(_))(x.reshape(nbatches, -1))
    return X.reshape(*shape_prefix, D, D)

def trilv(X):
    *shape_prefix, D = X.shape[:-1]
    nbatches = np.prod(shape_prefix, dtype=np.int64)
    x = vmap(lambda _: _[jnp.tril_indices(D)])(X.reshape(nbatches, D, D))
    return x.reshape(*shape_prefix, -1)