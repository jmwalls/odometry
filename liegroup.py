"""XXX
"""
import numpy as np

class SO2:
    """XXX
    """
    def __init__(self, *, x=1, y=0):
        """
        Note: The default options generate an identity SO2 object. However, most
        users should *not* directly initialize an SO2 object with non-default
        arguments (no normalization is performed here) and instead use the
        exponential map.
        """
        self._x = x
        self._y = y

    @classmethod
    def exp(cls, theta):
        """XXX
        """
        return cls(x=np.cos(theta), y=np.sin(theta))

    @staticmethod
    def hat(theta):
        """XXX
        """
        return np.array([[0.0, -theta], [theta, 0.0]])

    def log(self):
        """XXX
        """
        return np.arctan2(self._y, self._x)

    def adj(self):
        """XXX
        """
        return 1.0

    def inverse(self):
        """XXX
        """
        return SO2(x=self._x, y=-self._y)

    def __matmul__(self, other):
        """XXX
        """
        # SO2 composition...
        if isinstance(other, SO2):
            x = self._x * other._x - self._y * other._y
            y = self._x * other._y + self._y * other._x
            squared_norm = x * x + y * y
            if squared_norm != 1.0:
                scale = 2.0 / (1 + squared_norm)
                return SO2(x=x * scale, y=y * scale)
            return SO2(x=x, y=y)
        # Group action applied to 2d points
        if isinstance(other, np.ndarray):
            return (self.matrix() @ other.T).T
        if isinstance(other, list):
            return self.matrix() @ other
        raise ValueError('Unrecognized type for composition!')

    def matrix(self):
        """XXX
        """
        return np.array([[self._x, -self._y], [self._y, self._x]])


class SE2:
    """XXX
    """
    EPSILON = 1e-10

    # XXX set tangent frame index

    def __init__(self, *, so2=SO2(), p=np.zeros(2)):
        self.so2 = so2
        self.p = p

    @classmethod
    def exp(cls, phi):
        """XXX
        """
        theta = phi[2]
        p = phi[:2]

        so2 = SO2.exp(theta)

        if np.abs(theta) < SE2.EPSILON:
            theta_sq = theta * theta
            sin_theta_by_theta = 1.0 - (1.0 / 6) * theta_sq
            one_minus_cos_theta_by_theta = 0.5 * theta - (1.0 / 24) * theta * theta_sq
        else:
            sin_theta_by_theta = so2._y / theta
            one_minus_cos_theta_by_theta = (1.0 - so2._x) / theta
        return cls(so2=so2,
                   p=np.array([sin_theta_by_theta * p[0] - one_minus_cos_theta_by_theta * p[1],
                               one_minus_cos_theta_by_theta * p[0] + sin_theta_by_theta * p[1]]))

    @staticmethod
    def hat(phi):
        """XXX
        """
        return np.vstack([np.hstack([SO2.hat(phi[2]),
                                     np.reshape(phi[:2], (2, 1))]),
                          np.zeros(3)])

    def log(self):
        """XXX
        """
        theta = self.so2.log()
        halftheta = 0.5 * theta

        real_minus_one = self.so2._x - 1.0
        if np.abs(real_minus_one) < SE2.EPSILON:
            halftheta_by_tan_halftheta = 1.0 - (1.0 / 12) * theta * theta
        else:
            halftheta_by_tan_halftheta = -(halftheta * self.so2._y) / real_minus_one

        V_inv = np.array([[halftheta_by_tan_halftheta, halftheta],
                          [-halftheta, halftheta_by_tan_halftheta]])
        return np.hstack([V_inv @ self.p, theta])

    def adj(self):
        """XXX
        """
        print()
        return np.vstack([np.hstack([self.so2.matrix(),
                                     [[self.p[1]], [-self.p[0]]]]),
                          [0.0, 0.0, 1.0]])

    def inverse(self):
        """XXX
        """
        so2_inv = self.so2.inverse()
        return SE2(so2=so2_inv, p=-(so2_inv @ self.p))

    def __matmul__(self, other):
        """XXX
        """
        # SE2 composition...
        if isinstance(other, SE2):
            return SE2(so2=self.so2 @ other.so2, p=self.p + self.so2 @ other.p)
        # Group action applied to 2d points...
        if isinstance(other, np.ndarray):
            return self.so2 @ other + self.p
        if isinstance(other, list):
            return self.so2 @ other + self.p

    def matrix(self):
        """XXX
        """
        return np.vstack([np.hstack([self.so2.matrix(), self.p.reshape(2, 1)]),
                          np.array([0, 0, 1])])
