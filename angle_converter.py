import numpy as np
import math as m


def angle_calculation(axis, distance, point):
    k = np.abs(point - axis)
    k = np.interp(k, [0, point], [0, point * 0.4])  # piksel u cm !!!
    tg_alfa = k / distance
    alfa_rad = np.arctan(tg_alfa)
    alfa = m.degrees(alfa_rad)
    if axis > point:
        angle = 90 + alfa
    elif axis < point:
        angle = 90 - alfa
    else:
        angle = 90
    return angle

