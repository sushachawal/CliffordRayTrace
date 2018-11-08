from clifford.tools.g3c import *
from clifford.tools.g3c.GAOnline import *
import numpy as np
from PIL import Image
import time

red = 'rgb(255, 0 , 0)'
blue = 'rgb(0, 0, 255)'
green = 'rgb(0,255, 0)'
yellow = 'rgb(255, 255, 0)'
magenta = 'rgb(255, 0, 255)'
cyan = 'rgb(0,255,255)'
black = 'rgb(0,0,0)'
dark_blue = 'rgb(8, 0, 84)'
db = [0.033, 0., 0.33]


class Sphere:
    def __init__(self, c, r, colour, specular, spec_k, amb, diffuse, reflection):
        self.object = new_sphere(c + r*e1, c + r*e2, c + r*e3, c - r*e1)
        self.colour = colour
        self.specular = specular
        self.spec_k = spec_k
        self.ambient = amb
        self.diffuse = diffuse
        self.reflection = reflection
        self.type = "Sphere"

    def getColour(self):
        return "rgb(%d, %d, %d)"% (int(self.colour[0]*255), int(self.colour[1]*255), int(self.colour[2]*255))


class Plane:
    def __init__(self, p1, p2, p3, colour, specular, spec_k, amb, diffuse, reflection):
        self.object = new_plane(p1, p2, p3)
        self.colour = colour
        self.specular = specular
        self.spec_k = spec_k
        self.ambient = amb
        self.diffuse = diffuse
        self.reflection = reflection
        self.type = "Plane"

    def getColour(self):
        return "rgb(%d, %d, %d)" % (int(self.colour[0]*255), int(self.colour[1]*255), int(self.colour[2]*255))


class Light:
    def __init__(self, position, colour):
        self.position = position
        self.colour = colour


def drawScene():
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
    for points in rect:
        sc.add_euc_point(RMVR(up(points)), cyan)

    #Draw screen rectangle

    top = new_point_pair(Ptl, Ptr)
    right = new_point_pair(Ptr, Pbr)
    bottom = new_point_pair(Pbr, Pbl)
    left = new_point_pair(Pbl, Ptl)
    diag = new_point_pair(Ptl, Pbr)
    sides = [top, right, bottom, left, diag]
    for side in sides:
        sc.add_point_pair(RMVR(side), dark_blue)

    tl = new_line(eo, Ptl)
    tr = new_line(eo, Ptr)
    bl = new_line(eo, Pbl)
    br = new_line(eo, Pbr)

    lines = [tl, tr, br, bl]
    for line in lines:
        sc.add_line(RMVR(line).normal(), dark_blue)
    for objects in scene:
        if objects.type == "Sphere":
            sc.add_sphere(objects.object, objects.getColour())
        else:
            sc.add_plane(objects.object, objects.getColour())
    for light in lights:
        l = light.position
        sc.add_euc_point(up(l), yellow)
        sc.add_sphere(new_sphere(l + e1, l+e2, l+e3, l-e1), yellow)

    print(sc)


def new_sphere(p1, p2, p3, p4):
    return unsign_sphere(normalised(up(p1) ^ up(p2) ^ up(p3) ^ up(p4)))


def new_plane(p1, p2, p3):
    return normalised(up(p1) ^ up(p2) ^ up(p3) ^ einf)


def new_line(p1, p2):
    return normalised(up(p1) ^ up(p2) ^ einf)


def new_point_pair(p1, p2):
    return normalised(up(p1) ^ up(p2))


def unsign_sphere(S):
    return normalised(S/(S.dual()|einf)[0])


@numba.njit
def val_pointofXSphere(ray_val, sphere_val, origin_val):
    B = meet_val(ray_val, sphere_val)
    if gmt_func(B,B)[0] > 0.000001:
        point_vals = val_point_pair_to_end_points(B)
        if imt_func(point_vals[0,:],origin_val)[0] > imt_func(point_vals[1,:],origin_val)[0]:
            return point_vals[0,:]
    output = np.zeros(32)
    output[0] = -1
    return output


def pointofXsphere(ray, sphere, origin):
    B = meet(ray, sphere)
    if (B**2)[0] > 0.000001:
        points = PointsFromPP(B)
        if(points[0] | origin)[0] > (points[1] | origin)[0]:
            return points[0]
    return None


@numba.njit
def val_pointofXplane(ray_val, plane_val, origin_val):
    pX = val_intersect_line_and_plane_to_point(ray_val, plane_val)
    if pX[0] == -1.:
        return pX
    new_line = omt_func(origin_val, omt_func(pX, ninf_val))
    if abs((gmt_func(new_line, new_line))[0]) < 0.00001:
        return np.array([-1.])
    if imt_func(ray_val, val_normalised(new_line))[0] > 0:
        return pX
    return np.array([-1.])


def pointofXplane(ray, plane, origin):
    p = val_pointofXplane(ray.value, plane.value, origin.value)
    if p[0] == -1.:
        return None
    return layout.MultiVector(value=p)


def cosangle_between_lines(l1, l2):
    return (l1 | l2)[0]


def getfattconf(inner_prod, a1, a2, a3):
    return min(1./(a1 + a2 * np.sqrt(-inner_prod) - a3*inner_prod), 1.)


def getfatt(d, a1, a2, a3):
    return min(1./(a1 + a2*d + a3*d*d), 1.)


def PointsFromPP(mv):
    return point_pair_to_end_points(mv)


def reflect_in_sphere(ray, sphere, pX):
        return normalised((pX|(sphere*ray*sphere))^einf)


def intersects(ray, scene, origin):
    dist = -np.finfo(float).max
    index = None
    pXfin = None
    for idx, obj in enumerate(scene):
        if obj.type == "Sphere":
            pX = val_pointofXSphere(ray.value, obj.object.value, origin.value)
        if obj.type == "Plane":
            pX = val_pointofXplane(ray.value, obj.object.value, origin.value)

        if pX[0] == -1.: continue
        if idx == 0:
            dist, index, pXfin = imt_func(pX, origin.value)[0] , idx , layout.MultiVector(value=pX)
            continue
        t = imt_func(pX, origin.value)[0]
        if(t > dist):
            dist, index, pXfin = t, idx, layout.MultiVector(value=pX)
    return pXfin, index


def trace_ray(ray, scene, origin, depth):
    pixel_col = np.zeros(3)
    pX, index = intersects(ray, scene, origin)
    if index is None:
        return background
    obj = scene[index]
    for light in lights:
        Satt = 1.
        upl_val = val_up(light.position.value)
        toL = layout.MultiVector(value=val_normalised(omt_func(omt_func(pX.value, upl_val), einf.value)))
        d = layout.MultiVector(value=imt_func(pX.value, upl_val))[0]

        if options['ambient']:
            pixel_col += ambient * obj.ambient * obj.colour

        if intersects(toL, scene[:index] + scene[index+1:], pX)[0] is not None:
            Satt *= 0.8
        if obj.type == "Sphere":
            reflected = -1.*reflect_in_sphere(ray, obj.object, pX)
        else:
            reflected = layout.MultiVector(value=(gmt_func(gmt_func(obj.object.value, ray.value), obj.object.value)))

        norm = normalised(reflected - ray)

        fatt = getfattconf(d, a1, a2, a3)

        if options['specular']:
            pixel_col += Satt * fatt * obj.specular * \
                         max(cosangle_between_lines(norm, normalised(toL-ray)), 0) ** obj.spec_k * light.colour

        if options['diffuse']:
            pixel_col += Satt * fatt * obj.diffuse * max(cosangle_between_lines(norm, toL), 0) * obj.colour

    if depth >= max_depth:
        return pixel_col
    pixel_col += obj.reflection * trace_ray(reflected, scene, pX, depth + 1) / ((depth + 1) ** 2)
    return pixel_col


def RMVR(mv):
    return apply_rotor(mv, MVR)


# Light position and color.
lights = []
L = -20.*e1 + 60.*e3 + 4.*e2
colour_light = np.ones(3)
lights.append(Light(L, colour_light))
L = 20.*e1 + 60.*e3 + 4.*e2
#colour_light = np.array([1., 0., 0.])
lights.append(Light(L, colour_light))


# Shading options
a1 = 0.02
a2 = 0.0
a3 = 0.002
w = 1600
h = 1200
options = {'ambient': True, 'specular': True, 'diffuse': True}
ambient = 0.5
k = 1.  # Magic constant to scale everything by the same amount!
max_depth = 2
background = np.zeros(3) # [66./520., 185./510., 244./510.]

# Add objects to the scene:
scene = []
scene.append(Sphere(-2.*e1 - 7.2*e2 + 4.*e3, 4., np.array([1., 0., 0.]), k*1., 10., k*0.7, k*1., k*0.1))
scene.append(Sphere(6.*e1 - 2.0*e2 + 4.*e3, 4., np.array([0., 0., 1.]), k*1., 10., k*0.7, k*1., k*0.1))
scene.append(Plane(20.*e2+ e1, 20.*e2, 21.*e2, np.array([0.8, 0.8, 0.8]), k*0.3, 100., k*0.5, k*1., k*0.6))

# Camera definitions
cam = 4.*e3 - 20.*e2
lookat = eo
upcam = up(cam)
f = 1.
xmax = 1.0
ymax = xmax*(h*1.0/w)

# No need to define the up vector since we're assuming it's e3 pre-transform.

start_time = time.time()

# Get all of the required initial transformations
optic_axis = new_line(cam, lookat)
original = new_line(eo, e2)
MVR = generate_translation_rotor(cam)*rotor_between_lines(original, optic_axis)
dTx = MVR*generate_translation_rotor((2*xmax/(w-1))*e1)*~MVR
dTy = MVR*generate_translation_rotor(-(2*ymax/(h-1))*e3)*~MVR

Ptl = f*1.0*e2 - e1*xmax + e3*ymax

drawScene()


def render():
    img = np.zeros((h, w, 3))
    initial = RMVR(up(Ptl))
    clipped = 0
    for i in range(0, w):
        if i % 10 == 0:
            print(i/w * 100, "%")
        point = initial
        line = normalised(upcam ^ initial ^ einf)
        for j in range(0, h):
            value = trace_ray(line, scene, upcam, 0)
            new_value = np.clip(value, 0, 1)
            if np.any(value > 1.) or np.any(value < 0.):
                clipped += 1
            img[j, i, :] = new_value * 255.
            point = apply_rotor(point, dTy)
            line = normalised(upcam ^ point ^ einf)

        initial = apply_rotor(initial, dTx)
    print("Total number of pixels clipped = %d" % clipped)
    return img


im1 = Image.fromarray(render().astype('uint8'), 'RGB')
im1.save('fig.png')


print("\n\n")
print("--- %s seconds ---" % (time.time() - start_time))
