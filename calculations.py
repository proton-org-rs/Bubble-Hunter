import numpy as np
import math as m


def angle_calculation(axis, distance, point):
    k = point - axis
    k = np.interp(k, [0, point], [0, ])  # ovde treba kolko je 960(540) piksela u cm
    tg_alfa = np.abs(k) / distance
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
x = input()
y = input()
#d je const koju mi namestamo
d = 50 #cm

angle1 = angle_calculation(x, d, 960) #levo/desno
angle2 = angle_calculation(y, d, 540) #gore/dole
