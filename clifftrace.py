from clifford.tools.g3c import *
from clifford.tools.g3c.GAOnline import *
from numpy import e, pi
import math
import matplotlib.pyplot as plt
red = 'rgb(255, 0 , 0)'
blue = 'rgb(0, 0, 255)'
green = 'rgb(0,255, 0)'
yellow = 'rgb(255, 255, 0)'
magenta = 'rgb(255, 0, 255)'
cyan = 'rgb(0,255,255)'
black = 'rgb(0,0,0)'

def unsign_sphere(S):
    return (S/S.dual()|einf).normal()

def pointofXsphere(ray, sphere):
        B = meet(ray, unsign_sphere(sphere))
        if(B**2 > 0.00000001):
            return PointsFromPP(mv)[0]
        else: return None
def PointsFromPP(mv):
    P = 0.5*(1+(1/math.sqrt(mv**2))*mv)
    temp = mv|einf
    return(-~P*temp*P , P*temp*~P)

def RMVR(mv):
    return (MVR*mv*~MVR)

#Pixel resolution
w = 800
h = 600

#Camera definitions
cam =  0.35*e3 - 1.0*e2
lookat = 0.0
f = 2
xmax = 1.0
ymax  = xmax*(h*1.0/w)
#No need to define the up vector since we're assuming it's e3 pre-transform.

#Get all of the required initial transformations
optic_axis = (up(cam) ^ up(lookat) ^einf).normal()
original = (eo^ up(e2) ^ einf).normal()
MVR = generate_translation_rotor(cam)*rotor_between_lines(original, optic_axis)
theta = 2*math.atan(ymax/f*1.0)
phi = 2*math.atan(xmax/f*1.0)
Rx = generate_rotation_rotor(phi/2.,e1, e2)
Ry = generate_rotation_rotor(theta/2.,e2, e3)

tl = Rx*Ry*original*~Ry*~Rx

dtheta = math.atan(2*ymax/(f*1.0*h))
dphi = math.atan(2*xmax/(f*1.0*w))
dRx = MVR*generate_rotation_rotor(dphi, e2,e1)*~MVR
dRy = MVR*generate_rotation_rotor(dtheta, e3,e2)*~MVR
initial = RMVR(tl)
# for i in range(0,w):
#     line = initial
#     for j in range(0, h):
#         #trace_ray()
#         sc = GAScene()
#         sc.add_line(line.normal(), black)
#         line = dRy*line*~dRy
#     initial = dRx*initial*~dRx


















Ptl = f*1.0*e2 - e1*xmax + e3*ymax
Ptr = Ptl + 2*e1*xmax
Pbl = Ptl - 2*e3*ymax
Pbr = Ptr - 2*e3*ymax
rect = [Ptl, Ptr, Pbr, Pbl]

sc = GAScene()
sc.add_line(original, red)
sc.add_line((MVR*original*~MVR).normal(), red)
sc.add_euc_point(up(cam), blue)
sc.add_euc_point(up(lookat), blue)
sc.add_euc_point(up(Ptl), red)
sc.add_euc_point(up(Ptr), green)
sc.add_euc_point(up(Pbl), yellow)
sc.add_euc_point(up(Pbr), blue)
for points in rect:
    sc.add_euc_point(RMVR(up(points)), cyan)

tr = (~Rx)**2 * tl * Rx**2
br = (~Ry)**2 * tr * Ry**2
bl = (~Ry)**2 * tl * Ry**2
sc.add_line(tl.normal(), red)
sc.add_line(tr.normal(),green)
sc.add_line(br.normal(), blue)
sc.add_line(bl.normal(), yellow)

lines = [tl, tr, br, bl]
for line in lines:
    sc.add_line(RMVR(line).normal(),magenta)

print(sc)
