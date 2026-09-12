"""
The world a ranging sensor measures: a circular room with things in it.

This is deliberately the simplest geometry that produces a *recognisable* scan. A
circular wall gives a continuous baseline, and circular obstacles cut arcs out of it.
Plotted in polar form the result looks like a real room, which matters more than it
sounds: the point of a Lidar model is that a human can look at the scan and say "that
is wrong", and a per-angle lookup table cannot be wrong in any visible way.

Everything here is in metres and radians. The RPLIDAR protocol works in millimetres and
degrees, but converting at the protocol boundary rather than scattering factors of 1000
through the maths keeps the geometry readable and the unit bugs in one place.

WHAT RAY CASTING BUYS
---------------------
Range comes out of actual intersection maths, so the scan responds correctly to things
a table cannot represent: moving the sensor off-centre skews the wall distance
sinusoidally, an obstacle occludes a wedge of wall whose width depends on how close it
is, and driving the robot changes all of it consistently. A driver test that rotates the
sensor and expects the obstacle bearing to rotate with it is checking something real.

WHAT IS NOT MODELLED
--------------------
No reflectivity, no beam width, no multipath, no noise. A surface either returns at its
exact geometric distance or is out of range. Real Lidar returns are noisy and drop out
on dark or glancing surfaces; if you need to test how a driver handles bad data, inject
it rather than expecting this to produce it.
"""

import math


class Obstacle:
    """
    A circular obstacle. Position is in metres, in room coordinates.
    """

    def __init__(self, x, y, radius, name='obstacle'):
        if radius <= 0:
            raise ValueError(f'obstacle radius must be positive, not {radius}')
        self.x = x
        self.y = y
        self.radius = radius
        self.name = name

    def __repr__(self):
        return f'<Obstacle {self.name} at ({self.x:.2f}, {self.y:.2f}) r={self.radius:.2f}>'


class Room:
    """
    A circular room centred on the origin, optionally containing obstacles.

    :param radius: wall radius in metres
    :param obstacles: iterable of :class:`Obstacle`

    The sensor does not have to be at the centre, and putting it off-centre is the more
    interesting case -- the wall range then varies with angle, which is what makes a scan
    look like a scan.
    """

    def __init__(self, radius=4.0, obstacles=(), name='room'):
        if radius <= 0:
            raise ValueError(f'room radius must be positive, not {radius}')
        self.radius = radius
        self.obstacles = list(obstacles)
        self.name = name

    def add_obstacle(self, x, y, radius, name='obstacle'):
        """Put a circular obstacle in the room and return it."""
        obstacle = Obstacle(x, y, radius, name=name)
        self.obstacles.append(obstacle)
        return obstacle

    def contains(self, x, y):
        """Whether a point is inside the walls and not inside an obstacle."""
        if math.hypot(x, y) >= self.radius:
            return False
        return not any(
            math.hypot(x - o.x, y - o.y) < o.radius for o in self.obstacles
        )

    def range_at(self, origin, bearing):
        """
        Distance from ``origin`` to the first surface along ``bearing``.

        :param origin: ``(x, y)`` in metres
        :param bearing: direction in radians, measured counter-clockwise from +x
        :returns: distance in metres, or None if the ray escapes

        A ray from inside a circle always meets the wall, so None only happens if the
        origin is outside the room -- which is a caller error rather than a sensor
        reading, and is reported as "no return" rather than guessed at.
        """
        ox, oy = origin
        dx, dy = math.cos(bearing), math.sin(bearing)

        best = self._wall_distance(ox, oy, dx, dy)
        for obstacle in self.obstacles:
            hit = self._circle_distance(ox, oy, dx, dy, obstacle.x, obstacle.y,
                                        obstacle.radius)
            if hit is not None and (best is None or hit < best):
                best = hit
        return best

    @staticmethod
    def _solve(fx, fy, dx, dy, radius):
        """
        Roots of |origin + t*direction - centre|^2 = radius^2.

        ``f`` is the origin relative to the circle centre. The direction is a unit
        vector, so the quadratic's leading coefficient is 1 and it reduces to this.
        """
        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - radius * radius
        discriminant = b * b - c
        if discriminant < 0:
            return None
        root = math.sqrt(discriminant)
        return -b - root, -b + root

    def _wall_distance(self, ox, oy, dx, dy):
        """
        Where the ray leaves the room.

        The origin is inside, so the two roots straddle zero and the exit is always the
        larger one. Taking the smaller would give a distance behind the sensor.
        """
        roots = self._solve(ox, oy, dx, dy, self.radius)
        if roots is None:
            return None
        far = roots[1]
        return far if far > 0 else None

    def _circle_distance(self, ox, oy, dx, dy, cx, cy, radius):
        """Where the ray first meets an obstacle, if it does so ahead of the sensor."""
        roots = self._solve(ox - cx, oy - cy, dx, dy, radius)
        if roots is None:
            return None
        near, far = roots
        if near > 0:
            return near
        # Origin is inside this obstacle; the surface ahead is the far root.
        return far if far > 0 else None

    def __repr__(self):
        return (f'<Room {self.name} r={self.radius:.2f}m '
                f'{len(self.obstacles)} obstacles>')
