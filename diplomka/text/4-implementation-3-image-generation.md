## Basepair image generation {#sec:impl-basepair-img}

In order to visualize how similar of different various basepairs are, we use a PyMOL [TODO] script to generate an image of any basepair of interest.

This is implemented in the `gen_contact_images.py` Python script.
The script loads PyMOL as a library, it can be executed directly using Python interpreter.
However, the global environment must be used, since PyMOL is not published in the [PyPI repository](https://pypi.org/) and thus is not included in the poetry virtualenv.
The Polars library must be also installed globally.

For efficiency, the script is written to process a batch of basepairs.
It will identify all PDB structures in a given Parquet table, load each structure and generate the image for each basepair in the structure.
Different PDB structures may be processed in parallel, if the `--threads=X` option is specified.

The following steps are applied for each basepair and can be used to replicate a similar view in an interactive PyMOL session. (The commands are slightly simplified for the sake of this publication and are rewritten in the PyMOL scripting language, unless noted otherwise.)

* Select the basepair of interest (the fragments in curly braces are placeholders)

```
select leftnt, %{pdbid} and (chain {chain1} and resi {nr1}{ins1})
select rightnt, %{pdbid} ({residue_selection(chain2, nt2, ins2, alt2)})
select pair, %leftnt or %rightnt
```

* Show only the selection, add hydrogens if missing, and use a silver color for carbon atoms

```
hide everything
h_add %pair
show sticks, %pair
/util.cba("gray70")
```

* We have two options for orienting the basepair on the image - either optimizing for best screen use or for a similar position of different basepairs.

    a) In the first case, we use PyMOL built-in `orient` command to optimize for screen use. It aligns the plane with the image, and then makes it as wide as possible.

        ```
        orient %pair not (name C2' or name C3' or name C4' or name C5' or name O2' or name O3' or name O4' or name O5' or name P or name OP1 or name OP2)
        ```

    b) In the second case, we align the sugar—base bond of the right nucleotide with the vertical axis. First, make sure that the right nucleotide plane is aligned with the image:

        ```
        orient %rightnt not (name C2' or name C3' or name C4' or name C5' or name O2' or name O3' or name O4' or name O5' or name P or name OP1 or name OP2)
        ```

        And then orient the C1'-N9 or C1'-N1 bond with the vertical axis:

        ```python
        # Python code:
        coord1 = transform_to_camera_space(cmd.get_coords("%rightnt and (name C1')")[0])
        coord2 = transform_to_camera_space(cmd.get_coords("%rightnt and (name N9)")[0])
        # rotate around z-axis such that coord1 (C1') is right above coord2 (N)
        angle = np.arctan2(coord2[0] - coord1[0], coord2[1] - coord1[1])
        cmd.turn("z", math.degrees(angle))
        ```

        Last, we make sure that the "right" nucleotide is indeed on the right side. If it is not, the image is rotated 180° along Y axis ("flipped upside down").

* Center the image on the screen, using the `zoom %pair` command.
* Highlight hydrogen bonds and label the interacting atoms. <!-- For each hydrogen bond, we apply (using `{atom1}` and `{atom2}` placeholders) -->
```python
# Python script TODO
for i, (atom1, atom2) in enumerate(hbonds):
    cmd.distance(f"pair_hbond_{i}", atom_sele(atom1), atom_sele(atom2), mode=0)
    cmd.hide("labels", f"pair_hbond_{i}")
    if label_atoms:
        labels.extend([atom1, atom2])

if len(labels) > 0:
    cmd.label("(" + " or ".join([
        "(" + atom_sele(a) + ")"
        for a in labels
    ]) + ")", "name")
```

```
set label_color, black
set label_bg_color = "white", %pair
set label_outline_color = white
set label_size = 25
set label_font_id = 7
```

* Optionally, show water molecules (lone oxygen atoms) interacting with both nucleotides, the default distance threshold is 3.6 Å

```
select nwaters (resn HOH within 3.6 of %rightnt) and (resn HOH within 3.6 of %leftnt)

distance pair_w_contacts, %pair, %nwaters, mode=2
hide labels, pair_w_contacts

show nb_spheres, %nwaters
color 0xff0000, %nwaters
```

* Optionally, show the basepair surroundings as semi-transparent gray lines

```
show sticks, %{pdbid} and not %pair
set_bond stick_radius, 0.07, %{pdbid} and not %pair and not resname HOH
set_bond stick_transparency, 0.4, %{pdbid} and not %pair and not resname HOH
clip slab, 12, %pair
color 0xeeeeee, %{pdbid} and not %pair and not resname HOH
```

### Image output

The resulting PNG image is simply saved using the `png` command.

For the web-based basepair browser, we afterwards transcode all images using lossy AVIF codec.
We generate images of various resolutions, reducing the typically used network bandwidth to about 1/5 to 1/10.
The transcoding of individual images is performed using [ImageMagick](https://imagemagick.org), executed in parallel using Ninja build system.
To run the conversion, the `gen-images.sh` script generates the `build.ninja` file, which is then executed using the `ninja` command.

<!-- -rw-r--r-- 1 exyi exyi  23K Mar 21 17:15 img/1duh/A_56-A_53-1080.avif
-rw-r--r-- 1 exyi exyi  32K Mar 20 11:32 img/1duh/A_56-A_53-1440.avif
-rw-r--r-- 1 exyi exyi  39K Mar 21 09:39 img/1duh/A_56-A_53-1440.webp
-rw-r--r-- 1 exyi exyi  16K Mar 22 19:57 img/1duh/A_56-A_53-450.avif
-rw-r--r-- 1 exyi exyi  23K Mar  1 10:45 img/1duh/A_56-A_53-450.webp
-rw-r--r-- 1 exyi exyi  16K Mar 22 10:34 img/1duh/A_56-A_53-720.avif
-rw-r--r-- 1 exyi exyi 1.5M Oct 29 12:13 img/1duh/A_56-A_53.mp4
-rw-r--r-- 1 exyi exyi 152K Oct 29 12:12 img/1duh/A_56-A_53.png
-rw-r--r-- 1 exyi exyi  50K Mar 30 15:05 img/1duh/A_56-A_53-rotX-1080.avif
-rw-r--r-- 1 exyi exyi  57K Mar 30 10:55 img/1duh/A_56-A_53-rotX-1440.avif
-rw-r--r-- 1 exyi exyi  33K Mar 30 18:24 img/1duh/A_56-A_53-rotX-720.avif
-rw-r--r-- 1 exyi exyi 325K Mar 29 14:05 img/1duh/A_56-A_53-rotX.png -->

### Movie output

The script also generate a movie of the rotating basepair, if the `--movie` option is specified.
The scene is set up using the following commands.

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
For the web application, we encode the movie into webm/VP9 codec, preserving the alpha (transparency) channel.
