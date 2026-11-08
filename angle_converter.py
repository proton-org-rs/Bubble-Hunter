import numpy as np
import math


def angle_calculation(axis, distance, point):
    k = np.abs(point - axis)
    k = np.interp(k, [0, point], [0, point * 0.4])  # ovde treba kolko je 960(540) piksela u cm
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


#program kamere:
x = int(input())
y = int(input())
#d je const koju mi namestamo
d = 50 #cm

angle1 = round(angle_calculation(x, d, 960)) #levo/desno
angle2 = round(angle_calculation(y, d, 540)) #gore/dole as m
