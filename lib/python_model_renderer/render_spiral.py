import numpy as np
from shapely.geometry import LineString, MultiPoint


def render_spiral_shapely(points, poly_line, output_shape=(256, 256)):
    m = MultiPoint(points)
    line = LineString(poly_line)
    correct_values = np.fromiter((i.distance(line) for i in m), count=len(m), dtype=float).reshape(output_shape)
    return correct_values


def numpy_squared_distance_to_point(P, poly_line):
    """
    f(t) = (1−t)A + tB − P
    t = [(P - A).(B - A)] / |B - A|^2
    """
    u = P - poly_line[:-1]
    v = poly_line[1:] - poly_line[:-1]
    dot = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]
    t = np.clip(dot / (v[:, 0]**2 + v[:, 1]**2), 0, 1)
    # sep = (1.0 - t) * A + t*B - P
    # sep = A - t*A + t*B - P
    # sep = t*(B - A) - (P - A)
    sep = (v.T * t).T - u
    return np.min(sep[:, 0]**2 + sep[:, 1]**2)


_npsdtp_vfunc = np.vectorize(
    numpy_squared_distance_to_point,
    signature='(d),(n,d)->()'
)


def render_spiral_numpy(points, poly_line, output_shape=(256, 256)):
    return np.sqrt(
        _npsdtp_vfunc(
            points, poly_line,
        )
    ).reshape(*output_shape)
