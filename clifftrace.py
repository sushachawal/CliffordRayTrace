from clifford.tools.g3c import *
from clifford.tools.g3c.GAOnline import *
import numpy as np
from numpy import e, pi
import math
import matplotlib.pyplot as plt
import time

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
    return unsign_sphere((up(p1) ^ up(p2) ^ up(p3) ^ up(p4)).normal())

def new_line(p1, p2):
    return (up(p1)^up(p2)^einf).normal()

def new_point_pair(p1, p2):
    return (up(p1)^up(p2)).normal()

def unsign_sphere(S):
    return (S/(S.dual()|einf)[0]).normal()

def pointofXsphere(ray, sphere, origin):
    B = meet(ray, sphere)
    if((B**2)[0] > 0.000001):
        points = PointsFromPP(B)
        if((points[0] | origin)[0] > (points[1] | origin)[0]):
            return points[0]
    return None

def cosangle_between_lines(l1, l2):
    return ((l1|l2)/(math.sqrt(abs((l1**2)[0]))*math.sqrt(abs((l2**2)[0]))))[0]

def PointsFromPP(mv):
    P = 0.5*(1+(1/math.sqrt((mv**2)[0]))*mv)
    temp = mv|einf
    return(normalise_n_minus_1(-~P*temp*P) , normalise_n_minus_1(P*temp*~P))

def reflect_in_sphere(ray, sphere, pX):
        return(((pX|(sphere*ray*sphere))^einf).normal())

def intersects(ray, scene, origin):
    dist = None
    index = None
    pXfin = None
    for idx, obj in enumerate(scene):
        pX = pointofXsphere(ray, obj.object, origin)
        if(pX is None): continue
        if(idx == 0):
            dist, index, pXfin = (pX | origin) , idx , pX
            continue
        t = pX | origin
        if(t > dist):
            dist, index, pXfin = t, idx, pX
    return pXfin, index

def trace_ray(ray, scene,origin, depth):
    pixel_col = np.zeros(3)
    pX, index = intersects(ray, scene, origin)
    if(index is None): return background
    if(depth > 1):
        print("Object intersection from reflected ray!")
        sc = GAScene()
        sc.add_line(ray.normal())
        print(sc)
    obj = scene[index]
    # sc = GAScene()
    toL = (pX ^up(L)^einf).normal()
    if(intersects(toL, scene[:index] + scene[index+1:], pX)[0] is not None):
        return pixel_col
    reflected = -1.*reflect_in_sphere(ray, obj.object, pX)
    norm = reflected - ray
    #sc.add_line(norm.normal())
    # sc.add_line(toL, green)
    # sc.add_line(ray, cyan)
    # sc.add_line((toL - ray).normal())
    #Norm is not consistent!!!! NEED TO SORT OUT SIGN CONSISTENCY
    #print(sc)
    if(options['ambient']):
        pixel_col += ambient*obj.ambient*obj.colour
    if(options['specular']):
        pixel_col += obj.specular * max(cosangle_between_lines(norm, toL-ray), 0) ** obj.spec_k * colour_light
    if(options['diffuse']):
        pixel_col += obj.diffuse * max(cosangle_between_lines(norm, toL), 0) * obj.colour
    if(depth == max_depth):
        return pixel_col
    pixel_col += obj.reflection * trace_ray(reflected, scene, pX, depth + 1)
    return pixel_col

def RMVR(mv):
    return (MVR*mv*~MVR)

# Light position and color.
L = e1 + 10.*e3 -10.*e2
colour_light = np.ones(3)
ambient = 1.
options = {'ambient': True, 'specular': True, 'diffuse': True}

#Define background colour
background = np.array([0., 154./255., 1.])
background = np.array([0., 0., 0.])

#add objects to the scene!
scene = []
scene.append(Sphere(-2.*e1 + 0.2*e2, 4., np.array([0., 0., 1.]), 1., 50., .05, 1., 1.))

#Pixel resolution
w = 400
h = 300
max_depth = 2

#Camera definitions
cam =  6.*e3 - 6.*e2
lookat = 0.0
upcam = up(cam)
f = 1.
xmax = 1.0
ymax  = xmax*(h*1.0/w)
#No need to define the up vector since we're assuming it's e3 pre-transform.

start_time = time.time()

#Get all of the required initial transformations
optic_axis = new_line(cam, lookat)
original = new_line(eo, e2)
MVR = generate_translation_rotor(cam)*rotor_between_lines(original, optic_axis)
dTx = MVR*generate_translation_rotor((2*xmax/(w-1))*e1)*~MVR
dTy = MVR*generate_translation_rotor(-(2*ymax/(h-1))*e3)*~MVR

Ptl = f*1.0*e2 - e1*xmax + e3*ymax

img = np.zeros((h, w, 3))
initial = RMVR(up(Ptl))
for i in range(0,w):
    if (i%10 == 0):
        print(i/w * 100 , "%")
    point = initial
    line = (upcam ^ initial ^ einf).normal()
    for j in range(0, h):
        img[j, i, :] = np.clip(trace_ray(line, scene, upcam, 0), 0, 1)
        point = dTy*point*~dTy
        line = (upcam ^ point ^ einf).normal()

    initial = dTx*initial*~dTx

plt.imsave('fig.png', img)

Ptr = Ptl + 2*e1*xmax
Pbl = Ptl - 2*e3*ymax
Pbr = Ptr - 2*e3*ymax
rect = [Ptl, Ptr, Pbr, Pbl]

sc = GAScene()

#Draw Camera transformation
sc.add_line(original, red)
sc.add_line((MVR*original*~MVR).normal(), red)
sc.add_euc_point(up(cam), blue)
sc.add_euc_point(up(lookat), blue)

#Draw screen corners
sc.add_euc_point(Ptl, red)
sc.add_euc_point(up(Ptr), green)
sc.add_euc_point(up(Pbl), yellow)
sc.add_euc_point(up(Pbr), blue)
for points in rect:
    sc.add_euc_point(RMVR(up(points)), cyan)

#Draw screen rectangle

top = new_point_pair(Ptl, Ptr)
right = new_point_pair(Ptr, Pbr)
bottom = new_point_pair(Pbr, Pbl)
left = new_point_pair(Pbl, Ptl)
diag = new_point_pair(Ptl, Pbr)
sc.add_point_pair(top, yellow)
sc.add_point_pair(right, yellow)
sc.add_point_pair(bottom, yellow)
sc.add_point_pair(left, yellow)
sc.add_point_pair(diag, yellow)
sides = [top, right, bottom, left, diag]
for side in sides:
    sc.add_point_pair(RMVR(side), yellow)

tl = new_line(eo, Ptl)
tr = new_line(eo, Ptr)
bl = new_line(eo, Pbl)
br = new_line(eo, Pbr)
sc.add_line(tl, red)
sc.add_line(tr, green)
sc.add_line(br, blue)
sc.add_line(bl, yellow)

lines = [tl, tr, br, bl]
for line in lines:
    sc.add_line(RMVR(line).normal(),magenta)
sc.add_sphere(scene[0].object, blue)
sc.add_euc_point(up(L), yellow)
sc.add_sphere(new_sphere(L + e1, L+e2, L+e3, L-e1), yellow)

print(sc)
print("\n\n")
print("--- %s seconds ---" % (time.time() - start_time))
