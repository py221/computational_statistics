"""Joint, marginal and conditional distributions, covariance and correlation.

Run:  python 02-distributions/joint_covariance.py
"""
import numpy as np

# Joint pmf P(X=x, Y=y). Rows index x, columns index y.
X_VALUES = np.array([0, 1, 2])
Y_VALUES = np.array([0, 1])
JOINT = np.array([
    [0.10, 0.20],
    [0.15, 0.15],
    [0.30, 0.10],
])


def marginals(joint):
    return joint.sum(axis=1), joint.sum(axis=0)  # P_X, P_Y


def conditional_x_given_y(joint, j):
    """P(X = x | Y = y_j) for every x."""
    return joint[:, j] / joint[:, j].sum()


def covariance(joint, xs, ys):
    px, py = marginals(joint)
    ex, ey = xs @ px, ys @ py
    exy = xs @ joint @ ys
    return exy - ex * ey


def correlation(joint, xs, ys):
    px, py = marginals(joint)
    sx = np.sqrt(xs**2 @ px - (xs @ px) ** 2)
    sy = np.sqrt(ys**2 @ py - (ys @ py) ** 2)
    return covariance(joint, xs, ys) / (sx * sy)


def is_independent(joint, atol=1e-12):
    px, py = marginals(joint)
    return np.allclose(joint, np.outer(px, py), atol=atol)


def zero_cov_but_dependent():
    """X symmetric on {-1, 0, 1}, Y = X^2: Cov(X, Y) = 0 yet Y is a function of X."""
    xs = np.array([-1, 0, 1])
    ys = np.array([0, 1])
    joint = np.array([
        [0.0, 1 / 3],   # x=-1 -> y=1
        [1 / 3, 0.0],   # x= 0 -> y=0
        [0.0, 1 / 3],   # x= 1 -> y=1
    ])
    return joint, xs, ys


def main():
    px, py = marginals(JOINT)
    print("P_X =", px, " P_Y =", py)
    print("P(X | Y=0) =", conditional_x_given_y(JOINT, 0))
    cond_mean = X_VALUES @ conditional_x_given_y(JOINT, 0)
    print("E[X | Y=0] =", cond_mean, " E[X] =", X_VALUES @ px)
    print("Cov(X, Y) =", round(covariance(JOINT, X_VALUES, Y_VALUES), 4))
    print("rho       =", round(correlation(JOINT, X_VALUES, Y_VALUES), 4))
    print("independent?", is_independent(JOINT))

    joint, xs, ys = zero_cov_but_dependent()
    print("\nY = X^2 example:")
    print("Cov =", round(covariance(joint, xs, ys), 12), " independent?", is_independent(joint))


if __name__ == "__main__":
    main()
