# CGARayCast

## Overview:
This project uses [Conformal Geometric Algebra (CGA)][cga] to build a ray tracing engine with the [clifford][clifford] library with the following functionality:
* Recursive tracing of reflected rays up to a specified max depth
* A Blinn-Phong lighting model
* Shadows
* Spherical objects **only**
* A single point light source

<figure>
<img src="https://github.com/sushachawal/CliffordRayTrace/blob/master/Reflection.png?raw=true" alt="drawing" width="100%"/>
<figcaption style = "text-align: center"><b>An example rendering (Render time ~70s)</b></figcaption>
</figure>

## Setup
### Package requirements:

The script requires clifford, NumPy, matplotlib, 

[cga]: https://en.wikipedia.org/wiki/Conformal_geometric_algebra
[clifford]: https://github.com/pygae/clifford
