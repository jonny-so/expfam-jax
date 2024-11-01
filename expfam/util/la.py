import jax
import jax.numpy as jnp
import numpy as np

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