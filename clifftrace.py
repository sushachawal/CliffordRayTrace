from clifford.tools.g3c import *
from clifford.tools.g3c.GAOnline import *
import numpy as np
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

class Sphere:
    def __init__(self, c, r, colour, specular, spec_k, ambient, diffuse, reflection):
        self.object = new_sphere(c + r*e1, c + r*e2, c + r*e3, c - r*e1)
        self.colour = colour
        self.specular = specular
        self.spec_k = spec_k
        self.ambient = ambient
        self.diffuse = diffuse
        self.reflection = reflection

def new_sphere(p1, p2, p3, p4):
    return (up(p1) ^ up(p2) ^ up(p3) ^ up(p4)).normal()

def new_line(p1, p2):
    return (up(p1)^up(p2)^einf).normal()

def new_point_pair(p1, p2):
    return (up(p1)^up(p2)).normal()

def unsign_sphere(S):
    return (S/(S.dual()|einf)).normal()

def pointofXsphere(ray, sphere):
    scTemp = GAScene()
    scTemp.add_euc_point(upcam, green)
    scTemp.add_line(ray, red)
    scTemp.add_sphere(sphere, blue)
    print(scTemp)
    B = meet(ray, unsign_sphere(sphere))
    if((B**2)[0] > 0.000001):
        points = PointsFromPP(B)
        if((points[0] | upcam)[0] > (points[1] | upcam)[0]):
            return points[0]
    return None

def cosangle_between_lines(l1, l2):
    return (l1|l2)/(math.sqrt(abs((l1**2)[0]))*math.sqrt(abs((l2**2)[0])))[0]

def PointsFromPP(mv):
    P = 0.5*(1+(1/math.sqrt(mv**2))*mv)
    temp = mv|einf
    return(normalise_n_minus_1(-~P*temp*P) , normalise_n_minus_1(P*temp*~P))

def reflect_in_sphere(ray, sphere, pX):
        return(((pX|(sphere*ray*sphere))^einf).normal())

def intersects(ray, scene):
    dist = None
    index = None
    pXfin = None
    for idx, obj in enumerate(scene):
        pX = pointofXsphere(ray, obj.object)
        if(pX is None): continue
        if(idx == 0):
            dist, index = (pX | upcam) , idx
            continue
        t = pX | upcam
        if(t > dist):
            dist, index, pXfin = t, idx, pX
    return pXfin, index

def trace_ray(ray, scene, depth):
    pixel_col = np.zeros(3)
    pX, index = intersects(ray, scene)
    if(index is None): return background
    obj = scene[index]
    toL = new_line(pX, L)
    if(intersects(toL, scene[:index] + scene[index+1:])[0] is None):
        return pixel_col
    reflected = reflect_in_sphere(ray, obj.object)
    norm = reflected - ray
    pixel_col += ambient*obj.ambient
    pixel_col += obj.specular * max(cosangle_between_lines(norm, reflected), 0) ** obj.spec_k * colour_light
    pixel_col += obj.diffuse * max(cosangle_between_lines(norm, toL), 0) * obj.colour
    if(depth == max_depth):
        print("Found something and traced ray!")
        return pixel_col
    pixel_col += obj.reflection * trace_ray(reflected, scene, depth + 1)
    return pixel_col

def RMVR(mv):
    return (MVR*mv*~MVR)

# Light position and color.
L = 5.*e1 + 5.*e3 + 10.*e2
colour_light = np.ones(3)

#Define background colour
background = np.zeros(3)

#add objects to the scene!
scene = []
scene.append(Sphere(0.5*e1 + 0.2*e2, 4., [0., 0., 1.], 1., 50, .05, 1., 1.))

#Pixel resolution
w = 80
h = 60
max_depth = 2

#Camera definitions
cam =  6.*e3 - 6.*e2
lookat = 0.0
upcam = up(cam)
f = 0.1
xmax = 1.0
ymax  = xmax*(h*1.0/w)
#No need to define the up vector since we're assuming it's e3 pre-transform.

#Get all of the required initial transformations
optic_axis = new_line(cam, lookat)
original = new_line(eo, e2)
MVR = generate_translation_rotor(cam)*rotor_between_lines(original, optic_axis)
dTx = MVR*generate_translation_rotor((xmax/(w*1.0))*e1)*~MVR
dTy = MVR*generate_translation_rotor(-(ymax/(h*1.0))*e3)*~MVR

Ptl = f*1.0*e2 - e1*xmax + e3*ymax

img = np.zeros((w, h, 3))
initial = RMVR(up(Ptl))
# for i in range(0,w):
#     if (i%10 == 0):
#         print(i/w * 100 , "%")
#     point = initial
#     line = (upcam ^ initial ^ einf).normal()
#     for j in range(0, h):
#         img[i, j, :] = np.clip(trace_ray(line, scene, 0), 0, 1)
#         line = (upcam ^ (dTy*point*~dTy) ^ einf).normal()
#     initial = dTx*initial*~dTx
#
# plt.imsave('fig.png', img)


theta = 2*math.atan(ymax/f*1.0)
phi = 2*math.atan(xmax/f*1.0)
Rx = generate_rotation_rotor(phi/2.,e1, e2)
Ry = generate_rotation_rotor(theta/2.,e2, e3)
tl = Rx*Ry*original*~Ry*~Rx

Ptr = Ptl + 2*e1*xmax
Pbl = Ptl - 2*e3*ymax
Pbr = Ptr - 2*e3*ymax
rect = [Ptl, Ptr, Pbr, Pbl]

sc = GAScene()
sc.add_line(original, red)
sc.add_line((MVR*original*~MVR).normal(), red)
sc.add_euc_point(up(cam), blue)
sc.add_euc_point(up(lookat), blue)
sc.add_euc_point(Ptl, red)
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
sc.add_sphere(scene[0].object, blue)
print(sc)
