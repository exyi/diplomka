## Calculation of the proposed metrics

In section -@sec:basepair-metrics, we have informally described the basepairing measures we experimented with.
The section is dedicated to the exact definitions, including simplified Python code.

### Hydrogen bond lengths and angles

Let us say we have variables `res1` and `res2` with the residues, assuming a BioPython API (the `coord` attribute contains 3-element numpy array)
Moreover, we have a hydrogen bond definition in the `hbond` variable, which includes the atom names as its attributes.

The length is a distance between the interacting heavy atoms.
The donor and acceptor angles require an additional third atom, which is defined ahead of time.


```python
def angle(a, b, c):
    return math.degrees(np.arccos(
        np.dot(a - b, c - b) / (np.linalg.norm(a - b) * np.linalg.norm(c - b))
    ))
donor = res1.get_atom(hbond.donor)
acceptor = res2.get_atom(hbond.acceptor)
donor_neighbor = res1.get_atom(hbond.donor_neighbor)
acceptor_neighbor = res1.get_atom(hbond.acceptor_neighbor)

length = np.linalg.norm(donor.coord - acceptor.coord)
donor_angle = angle(donor_neighbor, donor, acceptor)
acceptor_angle = angle(acceptor_neighbor, acceptor, donor)
```

### Hydrogen bond planarity

The second set of metrics requires determination of the base planes, represented as the translation and an orthonormal basis of a new coordinate system.
We have mentioned that we are looking for optimal plane by least squared distance, but it is important to note that the distance L2, not the distance along the Y coordinate.
This makes the procedure more similar to Principal Component Analysis (PCA) or Kabsch algorithm ("RMSD alignment") than to linear regression.
The code is surprisingly <s>simple</s> short, an explanation of [the math can be found at](https://math.stackexchange.com/q/99317).
The first two basis vectors form the plane, while last one is the normal vector orthogonal to the plane.
<!-- We also define a projection function, which will be useful in the next step. -->

```python
atoms = ... # 3xN array - base atoms from residue
origin = np.mean(atoms, axis=0)
plane_basis, _, _ = np.linalg.svd((atoms - [ origin ]).T)
plane_normal = plane_basis[:, 2]
```
<!--projection_matrix = plane_basis @ plane_basis.T

# def plane_projection(point: np.ndarray) -> np.ndarray:
#     tpoint = point - origin
#     return tpoint @ projection_matrix + origin-->

We can now calculate the dot product of the H-bond vector and the normal, giving the cosine of their angle.
The angle to the plane is the same as angle to the normal, except shifted from the $0 \cdot 180$ range to $-90 \cdot 90$ (in other words, arcus sine of the dot product, instead of the arcus cosine).

```python
vector = res1.get_atom(hbond.donor).coord - res2.get_atom(hbond.acceptor).coord
cos_distance = np.dot(vector / np.linalg.norm(vector), plane_normal)
bond_plane_angle = 90 - math.degrees(math.acos(cos_distance))
```

### Plane to plane comparison

In this section, we calculate the overall **Coplanarity angle**, and the **Edge to plane distance** with the **Edge to plane angle**.
The **Coplanarity angle** is trivial, and the **Edge to plane angle** is calculated similarly to the **H-Bond to plane angle** above --
Instead of the hydrogen bond atoms, we take first and last atom of the edge.
The following code assumes that the `edge1` list contains the pairing edge atom coordinates of the first residue.


```python
# The overall angle
coplanarity_angle = math.degrees(math.arccos(np.dot(plane_normal1, plane_normal2)))

vector = edge1[0] - edge1[-1]
vector /= np.linalg.norm(vector)
egde1_to_plane_angle = math.degrees(math.asin(np.dot(vector, plane_normal2)))
```

The **Edge to plane angle** requires projecting the atoms onto the plane

```python
def plane_projection(basis, point: np.ndarray) -> np.ndarray:
    projection_matrix = basis @ basis.T
    tpoint = point - origin
    return tpoint @ projection_matrix + origin

edge_to_plane_distance = min(np.linalg.norm(plane_basis2, e - plane_projection(e)) for e in edge1)
```

### Relative base rotation

First, we need the coordinate system defined in @sec:basepair-metrics-ypr (and @fig:MMB_reference_frame-purinepluspyrimidine).
Similarly to the base plane, the coordinate system is represented by the origin position and its orthonormal basis (the same as rotation matrix).

TODO zkopčit kód z toho mailu asi

For the calculation of Yaw/Pitch/Roll angles, we use the [SciPy](https://doi.org/10.1038/s41592-019-0686-2) `Rotation` class.
The [`as_euler` method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html#r72d546869407-1){.link-no-footnote} can produce any kind of Euler angles, and the `"ZYX"` argument specifies that we want ZYX intrinsic angles.

```
```
