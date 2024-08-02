## Basepair Image Generation {#sec:impl-basepair-img}

In order to visualize how similar or different various basepairs are, we use a [PyMOL](https://github.com/schrodinger/pymol-open-source) Python API to generate images of all basepairs of interest.

This is implemented in the `gen_contact_images.py` Python script.
The script loads PyMOL as a library; it can be executed directly using the Python interpreter.
However, since PyMOL is not published on the PyPI repository (the official Python package index), it must be installed globally
The Polars library must be also installed globally, other dependencies are unnecessary for this script.

For efficiency, the script is written to process basepairs in batches.
In a given Parquet table, it identifies all PDB structures, loads each structure and generates the image for each basepair in the structure.
Different PDB structures are processed in parallel if the `--threads=X` option is specified.

The following steps are applied for each basepair and can be used to replicate a similar view in an interactive PyMOL session.
The commands are rewritten into the PyMOL scripting language and slightly simplified for publication.

1. Select the basepair of interest (the fragments in curly braces are placeholders)

    Listing: PyMOL script to select a given basepair {#lst:pymol-select-pair}

    ```
    select leftnt, {pdbid} and (chain {chain1} and resi {nr1}{ins1})
    select rightnt, {pdbid} and (chain {chain2} and resi {nr2}{ins2})
    select pair, %leftnt or %rightnt
    ```

2. Show only the selection, add hydrogens if missing, and use a silver color for carbon atoms

    Listing: PyMOL script to set up the desired visual representation {#lst:pymol-pair-visual}

    ```
    hide everything
    h_add %pair
    show sticks, %pair
    /util.cba("gray70")
    ```

3. We have two options for orienting the basepair on the image --- either optimizing for best screen use or for a similar position of different basepairs.

    a) In the first case, we use PyMOL built-in `orient` command to optimize for screen use. It aligns the plane with the image, and makes it as wide as possible.

        Listing: Alight the nucleotide base with the screen (**X**, **Y** axes) {#lst:pymol-orient-pair}

        ```
        orient %pair not (name C2' or name C3' or name C4' or
            name C5' or name O2' or name O3' or name O4' or
            name O5' or name P or name OP1 or name OP2)
        ```

    b) In the second case, we align the sugar-base bond of the right nucleotide with the vertical axis.
        First, make sure that the right nucleotide plane is aligned with the screen using the code from @lst:pymol-orient-pair, only substituting `%pair` for `%rightnt`.
        Then we proceed by orienting the glycosidic bond (C1'-N9/N1) with the vertical axis:

        Listing: PyMOL Python code to orient the glycosidic bond with **Y** axis by a single **Z** rotation {#lst:py-orient-glycobond}

        ```python
        def transform_to_camera_space(coords):
            coords = np.array(coords)
            view = cmd.get_view()
            matrix = np.array(view[0:9]).reshape((3, 3))
            camera_center = np.array(view[9:12])
            model_center = np.array(view[12:15])

            camera_coords = camera_center +
                np.dot(matrix.T, coords - model_center)
            return camera_coords

        coord1 = transform_to_camera_space(
            cmd.get_coords("%rightnt and (name C1')")[0])
        coord2 = transform_to_camera_space(
            cmd.get_coords("%rightnt and (name N9)")[0])
        # rotate around z-axis such that coord1 (C1')
        # is right above coord2 (N)
        angle = np.arctan2(
            coord2[0] - coord1[0],
            coord2[1] - coord1[1])
        cmd.turn("z", math.degrees(angle))
        ```

        Last, we make sure that the "right" nucleotide is indeed on the right side. If it is not, the image is rotated 180° along the **Y** axis ("flipped left to right").

4. Center the image on the screen, using the `zoom %pair` command.
5. Highlight hydrogen bonds using the `distance` command and label the interacting atoms by their name (using `label {atom}, name` command). For each hydrogen bond, we apply the [script @lst:pymol-label-hbond]:

    Listing: PyMOL script highlighting a H-bond specified by substituting `{atom1}` and `{atom2}` for the two H-bond atoms {#lst:pymol-label-hbond}

    ```
    distance pair_hbond, (%leftnt and name "{atom1}"), (%rightnt and name "{atom2}")
    hide labels, pair_hbond
    label (%leftnt and name "{atom1}") or (%rightnt and name "{atom2}"), name

    set label_color, black
    set label_bg_color = "white", %pair
    set label_outline_color = white
    set label_size = 25
    set label_font_id = 7
    ```

6. Optionally, we show water molecules (lone oxygen atoms) interacting with both nucleotides at distance below 3.6 Å

    Listing: PyMOL script showing nearby (≤ 3.6 Å) water molecules {#lst:pymol-nearby-waters}

    ```
    select nwaters (resn HOH within 3.6 of %rightnt) and (resn HOH within 3.6 of %leftnt)

    distance pair_w_contacts, %pair, %nwaters, mode=2
    hide labels, pair_w_contacts

    show nb_spheres, %nwaters
    color 0xff0000, %nwaters
    ```

7. Optionally, we show the basepair surroundings as semi-transparent gray lines:

    Listing: PyMOL script showing nearby molecules as gray lines {#lst:pymol-gray-lines}

    ```
    show sticks, %{pdbid} and not %pair
    set_bond stick_radius, 0.07, %{pdbid} and not %pair and not resname HOH
    set_bond stick_transparency, 0.4, %{pdbid} and not %pair and not resname HOH
    clip slab, 12, %pair
    color 0xeeeeee, %{pdbid} and not %pair and not resname HOH
    ```

### Image output {#sec:impl-basepair-img-img}

The resulting PNG image is simply saved using the PyMOL `png` command.
For the web-based basepair browser, we afterwards transcode all images using lossy AVIF codec.
We generate images of various resolutions, reducing the typically used network bandwidth to about 1/5 to 1/10th.
The transcoding of individual images is performed using [ImageMagick](https://imagemagick.org), executed in parallel using the Ninja build system.
To run the conversion, the `gen-images.sh` script generates the `build.ninja` file, which is then executed by Ninja.

### Movie output {#sec:impl-basepair-img-movie}

If the `--movie` option is specified, the script also generates a movie of the rotating basepair.
The scene is set up using the following commands.

Listing: PyMOL script to set up the rotating basepair movie {#lst:pymol-rotation-movie}

```
mset 1, 150
mview store, 1
mview store, 150
turn x, 120
mview store, 50, power=1
turn x, 120
mview store, 100, power=1
bg_color white
set ray_shadow = 0
set antialias = 1
mpng "output-directory/", width=640, height=480
```

PyMOL produces a directory of numbered PNG images, one for each frame of the animation.
For the web application, we use ffmpeg to encode the movie into the VP9 codec, which is capable of preserving the alpha (transparency) channel.

### Asymmetric units {#sec:impl-basepair-img-asy}

PyMOL directly supports loading the biological assembly of a structure, [when the `assembly` option is set to a number](https://pymolwiki.org/index.php/Assembly).
The assembly can be later split into the individual asymmetric units using the `split_states` command, creating new objects with a numeric suffix.
For instance, if we want to load the `6ros` structure:

```
set assembly = 1
fetch 6ros
split_states %6ros
```

This gives us three objects --- `6ros`, `6ros_0001`, and `6ros_0002`.
We delete the original object (`6ros`), use `6ros_0002` if any symmetry operation is specified for the nucleotide, and `6ros_0001` otherwise.
We are not aware of a way to map the new objects onto the PDB symmetry codes, although it can be presumed that PyMOL keeps the ordering of the source mmCIF file.
Fortunately, this is rarely an issue, since only a few nucleic acid structures have more than two repetitions in the assembly and the order does not matter if we have only two states.
<!-- Since ignorance is bliss, we simply use the second object when no symmetry operation is specified -->


Some structures, like [`4Lnt`](https://www.rcsb.org/structure/4Lnt) have multiple biological assemblies.
If the `assembly` option is non-empty, only one of the assemblies is loaded into PyMOL.
Since we usually do not know which assembly a given basepair originates from, we only set `assembly=1` when necessary --- when a basepair with symmetry operation exists.
