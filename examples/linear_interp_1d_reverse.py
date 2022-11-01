import numpy as np

xs = np.array([0, 1])
ys = np.array([1.5, 3.5])

def interp(x):
    dy = np.diff(ys)
    dx = np.diff(xs)
    dydx = dy / dx
    return ys[0] + dydx * x

print(interp(0.5))