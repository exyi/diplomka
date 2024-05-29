## Calculation of the Proposed Basepair Parameters

In section [-@sec:basepair-metrics], we have informally described the basepairing measures we experimented with.
The section is dedicated to the exact definitions, including simplified Python code.

### Hydrogen bond lengths and angles

The code in @lst:code-calc-hb-distance-angle assumes the existence of variables `res1` and `res2` with the residues from the BioPython, or a similar library (the `coord` attribute contains a NumPy vector of size 3).
Moreover, we have a hydrogen bond definition in the `hbond` variable, which includes the atom names as its attributes.

The length is a distance between the interacting heavy atoms.
The donor and acceptor angles require an additional third atom, which is defined ahead of time.

Listing: H-bond heavy atom distance and angles {#lst:code-calc-hb-distance-angle}

```python
import math, numpy as np
import Bio.PDB.Residue.Residue as Residue

def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Angle between 3 points"""
    return math.degrees(np.arccos(
        np.dot(a - b, c - b) / (math.dist(a, b) * math.dist(c, b))
    ))

def hbond_geometry(res1: Residue, res2: Residue, hbond):
    donor = res1.get_atom(hbond.donor)
    acceptor = res2.get_atom(hbond.acceptor)
    donor_neighbor = res1.get_atom(hbond.donor_neighbor)
    acceptor_neighbor = res1.get_atom(hbond.acceptor_neighbor)

    length = math.dist(donor.coord, acceptor.coord)
    donor_angle = angle(donor_neighbor, donor, acceptor)
    acceptor_angle = angle(acceptor_neighbor, acceptor, donor)

    return length, donor_angle, acceptor_angle
```

### Hydrogen bond planarity

The second set of parameters requires determination of the base planes, represented as the translation and an orthonormal basis of a new coordinate system.
We have mentioned that we are looking for optimal plane by least squared distance, but it is important to note that we need to use euclidean (L2) distance, not the distance along the Y coordinate.
This makes the procedure more similar to Principal Component Analysis (PCA) or Kabsch algorithm ("RMSD alignment") than to linear regression.
The plane fitting implementation in @lst:code-calc-base-plane-fit uses [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition), a good explanation of [the math can be found in StackExchange answers](https://math.stackexchange.com/q/99317).
SVD returns decomposes a rectangular matrix into three matrices, the first of which is an orthogonal matrix.
The first two columns of the matrix are the plane basis, while last one is the normal vector orthogonal to the plane.
The other two matrices would allow us to get the atom position in the new vector space, but that is unnecessary for this algorithm.
<!-- We also define a projection function, which will be useful in the next step. -->

Listing: Fit a plane to the base atoms with SVD {#lst:code-calc-base-plane-fit}

```python
def fit_plane(res: Residue):
    # 3xN array - base atoms from the residue
    # select only base, not backbone atoms
    base_atoms = ... # omitted for brevity
    origin = np.mean(base_atoms, axis=0)
    # SVD on centered atoms
    plane_basis, _, _ = np.linalg.svd((base_atoms - [ origin ]).T)
    plane_normal = plane_basis[:, 2]

    return origin, plane_basis, plane_normal
```
<!--projection_matrix = plane_basis @ plane_basis.T

# def plane_projection(point: np.ndarray) -> np.ndarray:
#     tpoint = point - origin
#     return tpoint @ projection_matrix + origin-->

As shown in @lst:code-calc-bond-to-plane, we can now calculate the dot product of the H-bond vector and the plane normal, getting the cosine of their angle.
The angle to the plane is the same as angle to the normal, except shifted from the $0 \cdot 180$ range to $-90 \cdot 90$ (in other words, arcus sine of the dot product, instead of the arcus cosine).

Listing: H-bond to plane calculation {#lst:code-calc-bond-to-plane}

```python
def hbond_plane_angle(plane_normal, res1: Residue, res2: Residue, hbond):
    vector = res1.get_atom(hbond.donor).coord -
        res2.get_atom(hbond.acceptor).coord
    cos_distance = np.dot(vector / np.linalg.norm(vector), plane_normal)
    bond_plane_angle = 90 - math.degrees(math.acos(cos_distance))
```

### Plane to plane comparison

In this section, we calculate the overall **Coplanarity angle**, and the **Edge to plane distance** with the **Edge to plane angle**.
The **Coplanarity angle** is trivial, and the **Edge to plane angle** is calculated similarly to the **H-Bond to plane angle** above --
Instead of the hydrogen bond atoms, we take first and last atom of the edge.
The code in @lst:code-calc-edge-to-plane assumes that the `edge1` list contains the pairing edge atom coordinates of the first residue.

The **Edge to plane distance** is calculated by projecting the atom coordinates onto the plane and measuring their distance.

Listing: Edge to plane angle calculation {#lst:code-calc-edge-to-plane}

```python
def plane_angles(plane_basis1, plane_basis2)
    plane_normal1 = plane_basis1[:, 2]
    plane_normal2 = plane_basis2[:, 2]

    # The overall angle
    coplanarity_angle = math.degrees(math.arccos(np.dot(plane_normal1, plane_normal2)))

    # Vector between edge atoms of length 1
    vector = edge1[0] - edge1[-1]
    vector /= np.linalg.norm(vector)

    # Angle of that vector to the other plane
    egde1_to_plane_angle = math.degrees(math.asin(np.dot(vector, plane_normal2)))

    # Minimal distance from edge to the other plane
    edge1_to_plane_distance = min(math.dist(e, plane_projection(plane_basis2, e)) for e in edge1)

    return coplanarity_angle, egde1_to_plane_angle, edge1_to_plane_distance

def plane_projection(basis: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Projects a point onto the first two vectors (=plane) from basis"""
    basis = basis[:, :2]
    projection_matrix = basis @ basis.T
    tpoint = point - origin
    return tpoint @ projection_matrix + origin
```

### Relative base rotation

First, we need the coordinate system defined in @sec:basepair-metrics-ypr and illustrated in @fig:MMB_reference_frame-purinepluspyrimidine.
Similarly to the base plane, the coordinate system is represented by the origin position and its orthonormal basis (a rotation matrix).
In this case, three atoms — three points in space define the coordinate system, and it can therefore be computed directly, without least squares fitting.
First, we set the origin to **N1** for pyrimidines or **N9** for purines.
The first basis vector (**Y**) is the vector from **C1'** to **N1**/**N9**, normalized to length **1**.
The third atom, **C6** for pyrimidines and **C8** for purines, now uniquely determines the plane — we set the **X** coordinate such that **Z = 0** at **C6**/**C8**.

Listing: Fit a coordinate system rooted in N1/N9 {#lst:code-calc-fit-YPR-coord}

```python
def N_coordinate_system(res: Bio.PDB.Residue.Residue):
    c1 = res.get_atom("C1'", None)
    if res.get_atom("N9", None) is not None:
        n = res.get_atom("N9", None)
        c2 = res.get_atom("C8", None)
    else:
        n = res.get_atom("N1", None)
        c2 = res.get_atom("C6", None)
    # N1/N9 is origin
    translation = -n.coord
    # Y axis is aligned with C1'-N
    y = n.coord - c1.coord
    y /= np.linalg.norm(y)
    # X axis is aligned with N1-C2/N9-C8 (but perpendicular to y)
    x = n.coord - c2.coord
    x -= np.dot(y, x) * y # project X onto Y
    x /= np.linalg.norm(x)
    # Z axis is perpendicular to x and y
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    rotation = np.array([x, y, z]).T

    return TranslationThenRotation(translation, rotation)
```

For the calculation of yaw/pitch/roll angles, we use the [SciPy](https://doi.org/10.1038/s41592-019-0686-2) `Rotation` class, as shown in @lst:code-calc-YPR-angles.
The [`as_euler` method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html#r72d546869407-1){.link-no-footnote} can produce any kind of Euler angles, and the `"ZYX"` argument specifies that we want ZYX intrinsic angles.
Using the function from @lst:code-calc-fit-YPR-coord, we determine the coordinate systems of the two nucleotides, most importantly, the rotation matrices.
The point of interest is the difference between the two matrices.
To preserve the intuitive notion of the aircraft going from the first nucleotide's glycosidic bond to the second one, the second nucleotide is rotated 180° along the Z axis.
Without the rotation, both bases point toward the helix center, making the yaw angle larger the “straighter” the basepair is.


Listing: Get yaw, pitch, and roll between two nucleotides {#lst:code-calc-YPR-angles}

```python
from scipy.spatial.transform.Rotation import Rotation

def get_rotation_angles(res1: Residue, res2: Residue):
    cs1 = N_coordinate_system(res1)
    cs2 = N_coordinate_system(res2)
    # flip along Z to fly "from" res1 "into" res2 
    flip = np.diag([-1, -1, 1])
    matrix = cs1.rotation.T @ (cs2.rotation @ flip)
    rotation = Rotation.from_matrix(matrix)
    yaw, pitch, roll = rotation.as_euler("ZYX", degrees=True)
    return yaw, pitch, roll
```
