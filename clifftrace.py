from clifford.tools.g3c import *
from clifford.tools.g3c.GAOnline import *
import numpy as np
from PIL import Image
import time
from scipy.optimize import fsolve

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

class Circle:
    def __init__(self, p1, p2, p3, colour, specular, spec_k, amb, diffuse, reflection):
        self.object = -new_circle(p1, p2, p3)
        self.colour = colour
        self.specular = specular
        self.spec_k = spec_k
        self.ambient = amb
        self.diffuse = diffuse
        self.reflection = reflection
        self.type = "Circle"

    def getColour(self):
        return "rgb(%d, %d, %d)" % (int(self.colour[0]*255), int(self.colour[1]*255), int(self.colour[2]*255))

class Interp_Surface:
    def __init__(self, C1, C2, colour, specular, spec_k, amb, diffuse, reflection):
        self.first = C1
        self.second = C2
        self.colour = colour
        self.specular = specular
        self.spec_k = spec_k
        self.ambient = amb
        self.diffuse = diffuse
        self.reflection = reflection
        self.type = "Surface"

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
    # sc.add_line(original, red)
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
        elif objects.type == "Plane":
            sc.add_plane(objects.object, objects.getColour())
        elif objects.type == "Circle":
            sc.add_circle(objects.object, objects.getColour())
        else:
            col = objects.getColour()
            sc.add_circle(objects.first, col)
            sc.add_circle(objects.second, col)
            for circles in [interp_objects_root(objects.first, objects.second, alpha/100) for alpha in range(1,100,20)]:
                sc.add_circle(circles, col)

    for light in lights:
        l = light.position
        sc.add_euc_point(up(l), yellow)
        sc.add_sphere(new_sphere(l + e1, l+e2, l+e3, l-e1), yellow)

    print(sc)


def new_sphere(p1, p2, p3, p4):
    return unsign_sphere(normalised(up(p1) ^ up(p2) ^ up(p3) ^ up(p4)))


def new_circle(p1, p2, p3):
    return normalised(up(p1) ^ up(p2) ^ up(p3))


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
    new_line1 = omt_func(origin_val, omt_func(pX, ninf_val))
    if abs((gmt_func(new_line1, new_line1))[0]) < 0.00001:
        return np.array([-1.])
    if imt_func(ray_val, val_normalised(new_line1))[0] > 0:
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


def pointofXcircle(ray_val, circle_val, origin_val):
    m = meet_val(ray_val, circle_val)
    if (np.abs(m) <= 0.000001).all():
        return np.array([-1.])
    elif gmt_func(m, m)[0] <= 0.00001:
        return val_pointofXplane(ray_val, omt_func(circle_val, einf.value), origin_val)
    else:
        return np.array([-1.])


def pointofXsurface(L, C1, C2, origin):
    # Check if the ray hits the endpoints

    # Check each

    # Check both
    def rootfunc(alpha):
        return [(meet(interp_objects_root(C1,C2,alpha[0]), L) ** 2)[0], (meet(interp_objects_root(C1,C2,alpha[1]), L) ** 2)[0]]

    # alpha_left and alpha_right are static function variables.
    # Initialised at the end of this function to 0 and 1 respectively
    sol = fsolve(rootfunc, np.array([pointofXsurface.alpha_left,pointofXsurface.alpha_right]),full_output=True)
    zeros_crossing = sol[0]
    success = sol[2]

    # Check if it misses entirely
    if success != 1:
        print("No alpha found!")
        print(sol)
        return np.array([-1.]), None

    if (zeros_crossing[0] < 0 or zeros_crossing[0] > 1) and  (zeros_crossing[1] < 0 or zeros_crossing[1] > 1):
        print("Returned out of bounds alpha values of: (%f,%f)" %(zeros_crossing[0], zeros_crossing[1]))
        return np.array([-1.]), None

    print("Alpha values are: ")
    print(zeros_crossing)
    pointofXsurface.alpha_left = zeros_crossing[0]
    pointofXsurface.alpha_right = zeros_crossing[1]

    # Check if it is in plane
    if np.abs(zeros_crossing[0] - zeros_crossing[1]) < 0.00001:
        # Intersect as it it were a sphere
        C = interp_objects_root(C1, C2, zeros_crossing[0])
        S = (C * (C ^ einf).normal() * I5).normal()
        return val_pointofXSphere(L.value, unsign_sphere(S).value, origin.value), zeros_crossing[0]

    # Get intersection points
    plane1_val = val_normalised(omt_func(interp_objects_root(C1, C2, zeros_crossing[0]).value, einf.value))
    plane2_val = val_normalised(omt_func(interp_objects_root(C1, C2, zeros_crossing[1]).value, einf.value))

    p1_val = val_pointofXplane(L.value, plane1_val, origin.value)
    p2_val = val_pointofXplane(L.value, plane2_val, origin.value)

    if p1_val[0] == -1. and p2_val[0] == -1.:
        return np.array([-1.]), None

    if p2_val[0] == -1.:
        return p1_val, zeros_crossing[0]

    if p1_val[0] == -1.:
        return p2_val, zeros_crossing[1]

    if imt_func(p1_val, origin.value)[0] > imt_func(p2_val, origin.value)[0]:
        return p1_val, zeros_crossing[0]
    else:
        return p2_val, zeros_crossing[1]

pointofXsurface.alpha_left = 0
pointofXsurface.alpha_right = 1

def project_points_to_circle(point_list, circle):
    """
    Takes a load of point and projects them onto a circle
    """
    circle_plane = (circle^einf).normal()
    planar_points = project_points_to_plane(point_list,circle_plane)
    circle_points = project_points_to_sphere(planar_points, -circle*circle_plane*I5)
    return circle_points

def val_differentiateCircle(alpha, C1_val, C2_val):
    X_val = alpha*C1_val + (1-alpha) * C2_val
    phiSquared = -gmt_func(X_val, adjoint_func(X_val))
    phiSq0 = phiSquared[0]
    phiSq4 = layout.MultiVector(value=phiSquared)(4).value
    dotz = C1_val - C2_val
    dotphiSq0 = 2*alpha*gmt_func(C1_val,C1_val)[0] - 2*(1-alpha)*gmt_func(C2_val,C2_val)[0] + (1-2*alpha)*(gmt_func(C1_val,C2_val)+gmt_func(C1_val,C1_val))[0]
    dotphiSq4 = (1-2*alpha) * layout.MultiVector(value=gmt_func(C1_val,C2_val)+gmt_func(C1_val,C1_val))(4).value
    tempsqrt = np.sqrt(phiSq0 -  gmt_func(phiSq4, phiSq4)[0])
    dott = (dotphiSq0 + (2*(phiSq0*dotphiSq0) - gmt_func(phiSq4, dotphiSq4) -gmt_func(dotphiSq4, phiSq4))/tempsqrt)[0]
    t = dotphiSq0 + tempsqrt
    sqrt2t = np.sqrt(2*t)
    f = t/(sqrt2t)
    dotf = (3*dott)/(2*sqrt2t)
    g = phiSq4/(sqrt2t)
    dotg = (4*t*dotphiSq4 - dott *phiSq4)/(2*t*sqrt2t)
    k = (f*f - gmt_func(g,g)[0])
    dotk = (2*f*dotf - gmt_func(g,dotg) - gmt_func(dotg, g))[0]
    fminusg = -g
    fminusg[0] += f
    dotfminusdotg = -dotg
    dotfminusdotg[0] += f
    term1 = k*gmt_func(fminusg, dotz)
    term2 = k*gmt_func(dotfminusdotg, X_val)
    term3 = dotk*gmt_func(fminusg, X_val)
    return layout.MultiVector(value=val_normalised(term1 + term2 + term3))(3)



def get_normal(C1,C2,alpha,P):
    # Aplus = interp_objects_root(C1,C2,alpha+0.001)
    # Aminus = interp_objects_root(C1,C2,alpha-0.001)
    # A = interp_objects_root(C1,C2,alpha)
    # Pplus = project_points_to_circle([P], Aplus)[0]
    # Pminus = project_points_to_circle([P], Aminus)[0]
    # CA = (Pminus ^ P ^ Pplus).normal()
    # Tangent_CA = ((CA | P) ^ einf).normal()
    # Tangent_A = ((A | P) ^ einf).normal()
    # normal = layout.MultiVector(value = project_val((Tangent_A*Tangent_CA*I5).value, 3)).normal()
    dotC = val_differentiateCircle(alpha, C2.value, C1.value)
    C = interp_objects_root(C1, C2, alpha)
    omegaC = C*dotC
    dotP = P|omegaC
    LT = (dotP ^ P ^ einf).normal()
    LC = ((C|P)^einf).normal()
    normal = (LT*LC*I5)(3).normal()
    return normal

def reflect_in_surface(ray, object, pX, alpha):
    sc = GAScene()
    sc.add_euc_point(pX, blue)
    file.write(str(sc) + "\n")
    normal = get_normal(object.first, object.second, alpha, pX)
    return normalised(normal + ray)


def intersects(ray, scene, origin):
    dist = -np.finfo(float).max
    index = None
    pXfin = None
    alpha = None
    alphaFin = None
    for idx, obj in enumerate(scene):
        if obj.type == "Sphere":
            pX = val_pointofXSphere(ray.value, obj.object.value, origin.value)
        if obj.type == "Plane":
            pX = val_pointofXplane(ray.value, obj.object.value, origin.value)
        if obj.type == "Circle":
            pX = pointofXcircle(ray.value, obj.object.value, origin.value)
        if obj.type == "Surface":
            pX, alpha = pointofXsurface(ray, obj.first, obj.second, origin)

        if pX[0] == -1.: continue
        if idx == 0:
            dist, index, pXfin, alphaFin = imt_func(pX, origin.value)[0] , idx , layout.MultiVector(value=pX), alpha
            continue
        t = imt_func(pX, origin.value)[0]
        if(t > dist):
            dist, index, pXfin, alphaFin = t, idx, layout.MultiVector(value=pX), alpha
    return pXfin, index, alphaFin

file = open("intersection_points.txt", "a+")
def trace_ray(ray, scene, origin, depth):
    pixel_col = np.zeros(3)
    pX, index, alpha = intersects(ray, scene, origin)
    if index is None:
        return background
    obj = scene[index]
    sc = GAScene()
    if obj.type == "Sphere":
        sc.add_euc_point(pX)
    else:
        sc.add_euc_point(pX, green)
    file.write(str(sc) + "\n")
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
        elif obj.type == "Plane":
            reflected = layout.MultiVector(value=(gmt_func(gmt_func(obj.object.value, ray.value), obj.object.value)))
        elif obj.type == "Circle":
            reflected = layout.MultiVector(value=(gmt_func(gmt_func(val_normalised(
                omt_func(obj.object.value, einf.value)), ray.value), val_normalised(omt_func(obj.object.value,einf.value)))))
        else:
            reflected = reflect_in_surface(ray, obj, pX, alpha)
        if obj.type == "Surface":
            norm = get_normal(obj.first, obj.second, alpha, pX)
        else:
            norm = normalised(reflected - ray)


        # tmp_scene = GAScene()
        # tmp_scene.add_line(ray, red)
        # tmp_scene.add_line(norm, green)
        # tmp_scene.add_line(reflected, green)
        # print(tmp_scene)

        fatt = getfattconf(d, a1, a2, a3)

        if options['specular']:
            # if obj.type == "Circle":
            #     print(Satt * fatt * obj.specular * \
            #              max(cosangle_between_lines(norm, normalised(toL-ray)), 0) ** obj.spec_k * light.colour)
            pixel_col += Satt * fatt * obj.specular * \
                         max(cosangle_between_lines(norm, normalised(toL-ray)), 0) ** obj.spec_k * light.colour

        if options['diffuse']:
            pixel_col += Satt * fatt * obj.diffuse * max(cosangle_between_lines(norm, toL), 0) * obj.colour * light.colour

    if depth >= max_depth:
        return pixel_col
    pixel_col += obj.reflection * trace_ray(reflected, scene, pX, depth + 1) #/ ((depth + 1) ** 2)
    return pixel_col


def RMVR(mv):
    return apply_rotor(mv, MVR)

def render():
    img = np.zeros((h, w, 3))
    initial = RMVR(up(Ptl))
    clipped = 0
    for i in range(0, w):
        # if i % 1 == 0:
        #     print(i/w * 100, "%")
        point = initial
        line = normalised(upcam ^ initial ^ einf)
        for j in range(0, h):
            print("Pixel coords are; %d, %d" % (j, i))
            value = trace_ray(line, scene, upcam, 0)
            new_value = np.clip(value, 0, 1)
            if np.any(value > 1.) or np.any(value < 0.):
                clipped += 1
            img[j, i, :] = new_value * 255.
            point = apply_rotor(point, dTy)
            line = normalised(upcam ^ point ^ einf)

        initial = apply_rotor(initial, dTx)
    # print("Total number of pixels clipped = %d" % clipped)
    return img

if __name__ == "__main__":
    # Light position and color.
    lights = []
    L = -20.*e1 + 5.*e3 -10.*e2
    colour_light = np.ones(3)
    lights.append(Light(L, colour_light))
    L = 20.*e1 + 5.*e3 -10.*e2
    lights.append(Light(L, colour_light))


    # Shading options
    a1 = 0.02
    a2 = 0.0
    a3 = 0.002
    w = 40
    h = 40
    options = {'ambient': True, 'specular': True, 'diffuse': True}
    ambient = 0.3
    k = 1.  # Magic constant to scale everything by the same amount!
    max_depth = 2
    background = np.zeros(3) # [66./520., 185./510., 244./510.]

    # Add objects to the scene:
    scene = []
    # scene.append(
    #     Sphere(-2. * e1 - 7.2 * e2 + 4. * e3, 4., np.array([1., 0., 0.]), k * 1., 100., k * 1., k * 1., k * 0.))
    # for i in range(-1, 3):
    #     scene.append(
    #         Sphere(i*e1 - 7.2 * e2 + 4. * e3, 4., np.array([1., 0., 0.]), k * 1., 100., k * 1., k * 1.0, k * 0.0))
    # scene.append(
    #     Plane(20. * e2 + e1, 20. * e2, 21. * e2, np.array([0.8, 0.8, 0.8]), k * 1., 100., k * 1., k * 1., k * 0.1))
    #
    # scene.append(
    #     Circle(2*e3-10.*e2, 2*e3-5*e2, 2*e3-10*e1-10*e2, np.array([1., 0., 0.]), k * 20., 100., k * 0.01, k * 1., k * 0.))
    #
    # points = [2*e3-10.*e2, 2*e3-5*e2, 2*e3-10*e1-10*e2]
    # rotate = generate_rotation_rotor(np.pi/6, e2, e3)
    # rotate1 = generate_rotation_rotor(np.pi/2.7, e1, e2)
    # new_points = [apply_rotor(p + 10*e2 + 15*e1, rotate) for p in points]
    # scene.append(
    #     Circle(new_points[0], new_points[1], new_points[2], np.array([0.2, 0.2, 0.2]), k * 1., 100., k * 0.2, k * 0.5, k * 2.)
    # )
    #
    # scene.append(
    #     Sphere(1*e2 + 8*e3 +11*e1, 2, np.array([0., 0., 1.]), k * 1., 100., k * 0.01, k * 1., k * 0.)
    # )
    #
    # new_points = [apply_rotor(apply_rotor(p+ 3*e1 + 15*e2 +5*e3, rotate1), rotate) for p in points]
    #
    # scene.append(
    #     Circle(new_points[0], new_points[1], new_points[2], np.array([0., 1., 0.]), k * 1., 100., k * 0.01, k * 0.5, k * 0.))
    #
    # scene.append(
    #     Sphere(7*e3 + 25*e2 + 5*e1, 7., np.array([0.2, 0.2, 0.2]), k * 1., 100., k * 0.2, k * 1.0, k * 1.)
    # )

    C1 = normalised(up(-4*e3) ^ up(4*e3) ^ up(4*e2))

    C2 = normalised(up(5*e1-4*e3) ^ up(5*e1+4*e3) ^ up(5*e1+4*e2))



    scene.append(
        Interp_Surface(C2, C1, np.array([0., 0., 1.]), k * 1., 100., k * .5, k * 1., k * 0.)
    )


    # Camera definitions
    cam = - 10.*e2 + 1.*e1
    lookat = e1
    upcam = up(cam)
    f = 1.
    xmax = 1.0
    ymax = xmax*(h*1.0/w)

    # No need to define the up vector since we're assuming it's e3 pre-transform.

    start_time = time.time()

    # Get all of the required initial transformations
    optic_axis = new_line(cam, lookat)
    original = new_line(eo, e2)
    MVR = generate_translation_rotor(cam-lookat)*rotor_between_lines(original, optic_axis)
    dTx = MVR*generate_translation_rotor((2*xmax/(w-1))*e1)*~MVR
    dTy = MVR*generate_translation_rotor(-(2*ymax/(h-1))*e3)*~MVR

    Ptl = f*1.0*e2 - e1*xmax + e3*ymax

    # drawScene()

    im1 = Image.fromarray(render().astype('uint8'), 'RGB')
    im1.save('figtestLatestSmall.png')

    equator_circle = (C1 + C2).normal()

    interp_sphere = (equator_circle * (equator_circle ^ einf).normal() * I5).normal()

    scene = [Sphere(0, 0, np.array([0., 0., 1.]), k * 1., 100., k * 0.5, k * 1., k * 0.)]
    scene[0].object = unsign_sphere(interp_sphere)
    # print("\n\nNow drawing Sphere:\n\n")
    # drawScene()

    im1 = Image.fromarray(render().astype('uint8'), 'RGB')
    im1.save('figtestSphereSmall.png')

    print("\n\n")
    print("--- %s seconds ---" % (time.time() - start_time))


# # Shading options
# a1 = 0.02
# a2 = 0.0
# a3 = 0.002
# w = 800
# h = 600
# options = {'ambient': True, 'specular': True, 'diffuse': True}
# ambient = 0.3
# k = 1.  # Brightness factor
# max_depth = 2
# background = np.zeros(3)
#
# # Add objects to the scene:
# scene = []
# scene.append(Sphere(-2.*e1 - 7.2*e2 + 4.*e3, 4., np.array([1., 0., 0.]), k*1., 100., k*1., k*1., k*0.1))
# scene.append(Sphere(6.*e1 - 2.0*e2 + 4.*e3, 4., np.array([0., 0., 1.]), k*1., 100., k*1., k*1., k*0.1))
# scene.append(Plane(20.*e2+ e1, 20.*e2, 21.*e2, np.array([0.8, 0.8, 0.8]), k*1., 100., k*1., k*1., k*0.1))
#
# # Camera definitions
# cam = 4.*e3 - 20.*e2
# lookat = eo
# upcam = up(cam)
# f = 1.
# xmax = 1.0
# ymax = xmax*(h*1.0/w)
#
# # No need to define the up vector since we're assuming it's e3 pre-transform.
#
# start_time = time.time()
#
# # Get all of the required initial transformations
# optic_axis = new_line(cam, lookat)
# original = new_line(eo, e2)
# MVR = generate_translation_rotor(cam)*rotor_between_lines(original, optic_axis)
# dTx = MVR*generate_translation_rotor((2*xmax/(w-1))*e1)*~MVR
# dTy = MVR*generate_translation_rotor(-(2*ymax/(h-1))*e3)*~MVR
#
# Ptl = f*1.0*e2 - e1*xmax + e3*ymax
#
# images = []
# dR = generate_rotation_rotor(np.pi/4, e3, e1)
# L = -20.*e1 + 4.*e2
# for i in range(1,5):
#     lights = []
#     lights.append(Light(L, np.ones(3)))
#     tmp = Image.fromarray(render().astype('uint8'), 'RGB')
#     tmp.save('./ims/frame{:02d}.png'.format(i))
    # if i == 2:
    #     tmp.save('frame.png')
    # images.append(Image.fromarray(render().astype('uint8'), 'RGB'))

    # L = apply_rotor(L, dR)
#
# for i in range(0,20):
#     images.append(Image.open('./ims/frame{:02d}.gif'.format(i)))
#
# images[0].save('animation.gif',
#                save_all=True,
#                append_images=images[1:],
#                duration=100,
#                loop=0)
#
#
# drawScene(Ptl, MVR, cam, lookat, scene, lights)
#
# print("\n\n")
# print("--- %s seconds ---" % (time.time() - start_time))