import logging
import numpy as np
from math import sqrt

from tqdm.contrib.logging import logging_redirect_tqdm


def logging_debug(msg):
    with logging_redirect_tqdm():
        logging.debug(msg)


def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3 - x1, y3 - y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    u = np.clip(u, 0, 1)

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py
    return np.hypot(x - x3, y - y3)


def line_distance(line, point):
    """Euclidean distance between a point and a line (x_min, x_max, y_min,  y_max)"""
    endpoints = np.array(line)[[0, 2, 1, 3]]
    return point_to_segment_dist(*endpoints, *point)


# helper functions for minimum spanning circle

# Python3 program to find the minimum enclosing
# circle for N integer points in a 2-D plane


# Function to return the euclidean distance
# between two points
def dist(a, b):
    return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))

    # Function to check whether a point lies inside


# or on the boundaries of the circle
def is_inside(c, p):
    return dist(c[0], p) <= c[1]

    # The following two functions are the functions used


# To find the equation of the circle when three
# points are given.
def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations
    )


def point_along_circle(h, k, r):
    theta = np.random.random() * 2 * np.pi
    return h + np.cos(theta) * r, k + np.sin(theta) * r


# Helper method to get a circle defined by 3 points
def get_circle_center(bx, by, cx, cy):
    B = bx * bx + by * by
    C = cx * cx + cy * cy
    D = bx * cy - by * cx
    return [(cy * B - by * C) // (2 * D), (bx * C - cx * B) // (2 * D)]

    # Function to return a unique circle that intersects


# three points
def circle_frOm(A, B, C):
    I = get_circle_center(B[0] - A[0], B[1] - A[1], C[0] - A[0], C[1] - A[1])
    I[0] += A[0]
    I[1] += A[1]
    return [I, dist(I, A)]

    # Function to return the smallest circle


# that intersects 2 points
def circle_from(A, B):

    # Set the center to be the midpoint of A and B
    C = [(A[0] + B[0]) / 2.0, (A[1] + B[1]) / 2.0]

    # Set the radius to be half the distance AB
    return [C, dist(A, B) / 2.0]

    # Function to check whether a circle encloses the given points


def is_valid_circle(c, P):

    # Iterating through all the points to check
    # whether the points lie inside the circle or not
    for p in P:
        if not is_inside(c, p):
            return False
    return True


# Function to return find the minimum enclosing
# circle from the given set of points
def minimum_enclosing_circle(P):

    # To find the number of points
    n = len(P)

    if n == 0:
        return [[0, 0], 0]
    if n == 1:
        return [P[0], 0]

        # Set initial MEC to have infinity radius
    mec = [[0, 0], float("inf")]

    # Go over all pair of points
    for i in range(n):
        for j in range(i + 1, n):

            # Get the smallest circle that
            # intersects P[i] and P[j]
            tmp = circle_from(P[i], P[j])

            # Update MEC if tmp encloses all points
            # and has a smaller radius
            if tmp[1] < mec[1] and is_valid_circle(tmp, P):
                mec = tmp

                # Go over all triples of points
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):

                # Get the circle that intersects P[i], P[j], P[k]
                tmp = circle_frOm(P[i], P[j], P[k])

                # Update MEC if tmp encloses all points
                # and has smaller radius
                if tmp[1] < mec[1] and is_valid_circle(tmp, P):
                    mec = tmp

    return mec


def is_collision_with_circle(circle_x, circle_y, rad, x, y):

    # Compare radius of circle
    # with distance of its center
    # from given point
    if (x - circle_x) * (x - circle_x) + (y - circle_y) * (
        y - circle_y
    ) <= rad * rad:
        return True
    else:
        return False
