"""The parts of Blender's mathutils (Vector, Euler, Quaternion) the fight scripts use, in plain Python."""
import math
import numpy as np


class Vector:
    __slots__ = ("x", "y", "z")

    def __init__(self, v=(0.0, 0.0, 0.0)):
        self.x, self.y, self.z = (float(a) for a in v)

    def __iter__(self): return iter((self.x, self.y, self.z))
    def __len__(self): return 3
    def __getitem__(self, i): return (self.x, self.y, self.z)[i]
    def __setitem__(self, i, v): setattr(self, "xyz"[i], float(v))
    def __repr__(self): return f"Vector(({self.x:.4f}, {self.y:.4f}, {self.z:.4f}))"
    def __add__(self, o): return Vector((self.x + o[0], self.y + o[1], self.z + o[2]))
    __radd__ = __add__
    def __sub__(self, o): return Vector((self.x - o[0], self.y - o[1], self.z - o[2]))
    def __rsub__(self, o): return Vector((o[0] - self.x, o[1] - self.y, o[2] - self.z))
    def __mul__(self, k): return Vector((self.x * k, self.y * k, self.z * k))
    __rmul__ = __mul__
    def __truediv__(self, k): return Vector((self.x / k, self.y / k, self.z / k))
    def __neg__(self): return Vector((-self.x, -self.y, -self.z))
    def copy(self): return Vector(self)
    def dot(self, o): return self.x * o[0] + self.y * o[1] + self.z * o[2]
    def cross(self, o): return Vector(np.cross(tuple(self), tuple(o)))

    @property
    def length(self): return math.sqrt(self.dot(self))

    def normalized(self):
        L = self.length
        return self / L if L > 0 else Vector(self)

    def lerp(self, o, t): return self + (Vector(o) - self) * t

    def to_track_quat(self, track="Y", up="Z"):
        """A rotation that points the local `track` axis along this vector, with the local `up`
        axis turned as close to world +Z as it can (what the fight scripts rely on)."""
        return Quaternion.from_matrix(track_matrix(np.array(tuple(self)), track, up))


def track_matrix(vec, track, up):
    axes = {"X": 0, "Y": 1, "Z": 2}
    sign = -1.0 if track.startswith("-") else 1.0
    t, u = axes[track[-1]], axes[up[-1]]
    d = vec / (np.linalg.norm(vec) or 1.0) * sign
    world_up = np.array([0.0, 0.0, 1.0])
    w = world_up - d * d.dot(world_up)
    if np.linalg.norm(w) < 1e-6:
        w = np.array([0.0, 1.0, 0.0]) - d * d[1]
    w /= np.linalg.norm(w)
    M = np.zeros((3, 3)); M[:, t] = d; M[:, u] = w
    k = 3 - t - u
    M[:, k] = np.cross(M[:, t], M[:, u]) if (t + 1) % 3 == u else np.cross(M[:, u], M[:, t])
    return M


def euler_matrix(e):
    """Blender XYZ Euler -> 3x3 matrix."""
    cx, sx = math.cos(e[0]), math.sin(e[0]); cy, sy = math.cos(e[1]), math.sin(e[1]); cz, sz = math.cos(e[2]), math.sin(e[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Euler(tuple):
    def __new__(cls, v=(0.0, 0.0, 0.0), order="XYZ"): return super().__new__(cls, (float(a) for a in v))
    x = property(lambda s: s[0]); y = property(lambda s: s[1]); z = property(lambda s: s[2])
    def to_matrix(self): return euler_matrix(self)


class Quaternion:
    def __init__(self, a=(1.0, 0.0, 0.0, 0.0), angle=None):
        if angle is None:
            self.q = np.array(a, float)
        else:                                   # Quaternion(axis, angle)
            ax = np.array(a, float); ax /= np.linalg.norm(ax)
            self.q = np.concatenate([[math.cos(angle / 2)], ax * math.sin(angle / 2)])

    @staticmethod
    def from_matrix(M):
        tr = M[0, 0] + M[1, 1] + M[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1) * 2
            q = [0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = math.sqrt(1 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            q = [(M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
        elif M[1, 1] > M[2, 2]:
            s = math.sqrt(1 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            q = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s]
        else:
            s = math.sqrt(1 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            q = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s]
        return Quaternion(q)

    def to_matrix(self):
        w, x, y, z = self.q / np.linalg.norm(self.q)
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def __matmul__(self, o):
        w1, x1, y1, z1 = self.q; w2, x2, y2, z2 = o.q
        return Quaternion((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                           w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2))

    def to_euler(self, order="XYZ"):
        M = self.to_matrix()
        cy = math.hypot(M[0, 0], M[1, 0])
        if cy > 1e-6:
            return Euler((math.atan2(M[2, 1], M[2, 2]), math.atan2(-M[2, 0], cy), math.atan2(M[1, 0], M[0, 0])))
        return Euler((math.atan2(-M[1, 2], M[1, 1]), math.atan2(-M[2, 0], cy), 0.0))
