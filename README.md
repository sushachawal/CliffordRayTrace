# CGARayCast

## Overview:
This project uses [Conformal Geometric Algebra (CGA)][cga] to build a ray tracing engine with the [clifford][clifford] library with the following functionality:
* Recursive tracing of reflected rays up to a specified max depth
* A Blinn-Phong lighting model
* Shadows
* Spherical objects **only**
* A single point light source

<figure>
<img src="https://github.com/sushachawal/CliffordRayTrace/blob/master/Example.png?raw=true" alt="drawing" width="100%"/>
<figcaption style = "text-align: center"><b>An example rendering (Render time ~50s)</b></figcaption>
</figure>

## Usage:

Run the script with `python3 clifftrace.py`
### Package requirements:

The script requires the following non-standard python packages:
* [clifford][clifford]
* [NumPy][NumPy]
* [matplotlib][matplotlib]

### Interface with GAOnline:

The output image is saved in the working directory as `fig.png` but there is also a terminal output which allows the scene composition to be drawn in [GAOnline][GAOnline] so that it can be viewed interactively. An example terminal output is:
```
DrawLine((1.0^e245),rgb(255, 0 , 0));
DrawLine((0.70711^e245) - (0.70711^e345),rgb(255, 0 , 0));
DrawEucPoint(-(6.0^e2) + (6.0^e3) + (35.5^e4) + (36.5^e5),rgb(0, 0, 255));
...
```

Copy and paste the terminal output into the **box outlined in red** in the [GAOnline][GAOnline] example image shown below.

<figure>
<img src="https://github.com/sushachawal/CliffordRayTrace/blob/master/GAOnline.png?raw=true" alt="drawing" width="100%"/>
<figcaption style = "text-align: center"><b>An example view in GAOnline</b></figcaption>
</figure>

The output draws:

1. **Camera:** The position, optic axis, viewing screen and corner rays.
2. **Objects in the scene:** Object geometry and colour.
3. **Lighting:** Position will be drawn as yellow point inside a sphere.

[cga]: https://en.wikipedia.org/wiki/Conformal_geometric_algebra
[clifford]: https://github.com/pygae/clifford
[NumPy]: https://github.com/numpy/numpy
[matplotlib]: https://github.com/matplotlib/matplotlib
[GAOnline]: https://gaonline.herokuapp.com/
