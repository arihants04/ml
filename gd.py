import numpy as np

def gradient_descent(
     gradient, start, learn_rate, n_iter=100, tolerance=1e-3
 ):
  vector = start
  for _ in range(n_iter):
    diff = -learn_rate * gradient(vector)
    if np.all(np.abs(diff) <= tolerance):
      break
    vector += diff
  return vector

res = gradient_descent(
     gradient=lambda v: 2 * (v-2), start=5.0, learn_rate=0.2
)

print(res)

res = gradient_descent(
gradient=lambda v: 4 * v**3 - 10 * v - 3, start=3.5, learn_rate=0.05)

print(res)

res = gradient_descent(
    gradient=lambda v: np.array([2 * v[0], 4 * v[1]**3]),
    start=np.array([1.0, 1.0]), learn_rate=0.2, tolerance=1e-08
)
print(res)


res = gradient_descent(
    gradient=lambda v: 1 - 1 / v, start=2.5, learn_rate=0.5
)
print(res)

# the gradient for my situation
def ssr_gradient(x, y, w):
    res = w[0] + w[1] * x - y
    return res.mean(), (res*x).mean()  # .mean() is a method of np.ndarray

# generic method to minimize ANY convex function in the world
def gradient_descent(
     gradient, x, y, start, learn_rate=0.1, n_iter=50, tolerance=1e-06
 ):
  vector = start
  for _ in range(n_iter):
    diff = -learn_rate * np.array(gradient(x, y, vector))
    if np.all(np.abs(diff) <= tolerance):
      break
    vector += diff
  return vector



x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([5, 20, 14, 32, 22, 38])

res = gradient_descent(
    ssr_gradient, x, y, start=[0.5, 0.5], learn_rate=0.0008,
    n_iter=100000
)
print(res)