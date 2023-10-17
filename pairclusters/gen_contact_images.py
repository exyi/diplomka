import pymol
from pymol import cmd
import polars as pl
import os, sys
import pdb_utils
import numpy as np
import math
from dataclasses import dataclass
import shutil
import subprocess

def residue_selection(chain, nt, ins, alt):
    chain = str(chain)
    nt = str(nt).replace("-", "\\-")
    if ins and ins != ' ' and ins != '?':
        nt += str(ins)

    if alt and alt != ' ' and alt != '?':
        alt = f" alt {alt}"
    else:
        alt = ""
    
    return f"(chain {chain} and resi {nt}{alt})"

def load(file, pdbid):
    if file:
        cmd.load(file)
    else:
        if len(pdb_utils.pdb_cache_dirs) == 0:
            cmd.fetch(pdbid)
        else:
            file = pdb_utils.get_pdb_file(None, pdbid)
            print(file)
            cmd.load(file)

def transform_to_camera_space(coords):
    coords = np.array(coords)
    view = cmd.get_view()
    matrix = np.array(view[0:9]).reshape((3, 3))
    camera_center = np.array(view[9:12])
    model_center = np.array(view[12:15])

    camera_coords = camera_center + np.dot(matrix.T, coords - model_center)
    return camera_coords

def rotate_to_y_axis(bond1, bond2):
    coord1 = transform_to_camera_space(cmd.get_coords(bond1)[0])
    coord2 = transform_to_camera_space(cmd.get_coords(bond2)[0])
    # rotate around z-axis such that coord1 is right above coord2
    angle = np.arctan2(coord2[0] - coord1[0], coord2[1] - coord1[1])
    angle = angle / math.pi * 180
    # print("the bond coordinate is ", coord1, coord2)
    # print(f"1: {np.arctan2(coord1[1], coord1[0]) / math.pi * 180}, 2: {np.arctan2(coord2[1], coord2[0]) / math.pi * 180}")
    print("rotating by ", angle)
    cmd.turn("z", angle)
    # cmd.color("red", f"({bond1}) or ({bond2})")
    return abs(angle) > 0.1

def orient_nucleotide_as_main():
    if cmd.get_coords("%rightnt and (name N9)") is None:
        natom = "N1"
    else:
        natom = "N9"
    for i in range(1):
        # it does not converge instantly, no idea why
        rotate_to_y_axis("%rightnt and (name C1')", f"%rightnt and (name {natom})")

    # main nucleotide should be on the left
    cc1 = transform_to_camera_space(cmd.get_coords("%rightnt and (name C1')")[0])
    if cc1[0] > 0:
        cmd.turn("y", 180)
        cc2 = transform_to_camera_space(cmd.get_coords("%rightnt and (name C1')")[0])
        print(f"flipped {cc1} to {cc2}")
        rotate_to_y_axis("%rightnt and (name C1')", f"%rightnt and (name {natom})")

def orient_pair(pdbid, chain1, nt1, ins1, alt1, chain2, nt2, ins2, alt2, label_atoms: list[str],
    standard_orientation,
    grey_context=False):
    cmd.hide("everything", f"%{pdbid}")
    pair_selection =f"%{pdbid} and ({residue_selection(chain1, nt1, ins1, alt1)} or {residue_selection(chain2, nt2, ins2, alt2)})"
    print(pair_selection)
    cmd.select("pair", pair_selection)
    cmd.select("rightnt", f"%pair and ({residue_selection(chain1, nt1, ins1, alt1)})")
    # cmd.select("rightnt", f"%pair and ({residue_selection(chain2, nt2, ins2, alt2)})")
    cmd.show("sticks", "%pair")
    cmd.delete("pair_contacts")
    cmd.distance("pair_contacts", "%pair", "%pair", mode=2)
    cmd.hide("labels", "pair_contacts")
    cmd.util.cba("grey", f"%pair")
    if "all" in label_atoms:
        cmd.label("%pair", "name")
    elif len(label_atoms) > 0:
        cmd.label("%pair and (" + " or ".join([
            (
                f"({residue_selection(chain1, nt1, ins1, alt1)} and name \"{a[1:]}\")"
                if a.startswith("A") else
                f"({residue_selection(chain2, nt2, ins2, alt2)} and name \"{a[1:]}\")"
                if a.startswith("B") else
                f"name \"{a}\""
            )
            for a in label_atoms
        ]) + ")", "name")
    if standard_orientation:
        cmd.orient("%rightnt")
        orient_nucleotide_as_main()
    else:
        cmd.orient("%pair")
    cmd.zoom("%pair", 0)
    # cmd.center("%rightnt")
    cmd.set("label_color", "black")
    cmd.set("label_bg_color", "white", "%pair")
    cmd.set("label_outline_color", "white")
    cmd.set("label_size", 25)
    # cmd.set("label_position", (0, 0, 0))
    cmd.set("label_font_id", 7)
    if grey_context:
        cmd.show("sticks", f"%{pdbid} and not %pair")
        cmd.set_bond("stick_radius", 0.07, f"%{pdbid} and not %pair")
        cmd.set_bond("stick_transparency", 0.4, f"%{pdbid} and not %pair")
        cmd.clip("slab", 12, "%pair")
        cmd.color("0xeeeeee", f"%{pdbid} and not %pair")

@dataclass
class BPArgs:
    label_atoms: list[str]
    standard_orientation: bool
    movie: int
    incremental: bool

def make_pair_image(output_file, pdbid, chain1, nt1, ins1, alt1, chain2, nt2, ins2, alt2, bpargs: BPArgs):
    orient_pair(pdbid, chain1, nt1, ins1, alt1, chain2, nt2, ins2, alt2, bpargs.label_atoms, bpargs.standard_orientation)
    cmd.png(output_file, width=2560, height=1440, ray=1)
    print(f"Saved basepair image {output_file}")

def make_pair_rot_movie(output_file, pdbid, chain1, nt1, ins1, alt1, chain2, nt2, ins2, alt2, bpargs: BPArgs):
    length = bpargs.movie
    if length == 0:
        return

    orient_pair(pdbid, chain1, nt1, ins1, alt1, chain2, nt2, ins2, alt2, bpargs.label_atoms, bpargs.standard_orientation)
    try:
        cmd.mset("1", length)
        cmd.mview("store", 1)
        cmd.mview("store", length)
        cmd.turn("x", 120)
        cmd.mview("store", length // 3, power=1)
        cmd.turn("x", 120)
        cmd.mview("store", length - length // 3, power=1)
        cmd.bg_color("white")
        cmd.set("ray_shadow", 0)
        cmd.set("antialias", 1)

        use_ffmpeg_directly = True
        if output_file.endswith(".dir"):
            os.makedirs(output_file, exist_ok=True)
            cmd.mpng(output_file + "/", width=640, height=480)
        elif use_ffmpeg_directly:
            os.makedirs(output_file + ".pngdir", exist_ok=True)
            try:
                cmd.mpng(output_file + ".pngdir/", width=1280, height=720)
                if os.path.exists(output_file):
                    os.remove(output_file)
                ffmpeg_result = subprocess.run([
                    "ffmpeg",
                    "-framerate", "30",
                    "-pattern_type", "glob",
                    "-i", output_file + ".pngdir/*.png",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    output_file
                ], capture_output=True)
                if ffmpeg_result.returncode != 0:
                    print("ffmpeg failed")
                    print(ffmpeg_result.stdout.decode("utf-8"))
                    print(ffmpeg_result.stderr.decode("utf-8"))
            finally:
                shutil.rmtree(output_file + ".pngdir")
        else:
            from pymol import movie
            movie.produce(output_file, mode="draw", quality=96, width=1280, height=720, encoder="ffmpeg")
        print("movie produced")
    finally:
        cmd.mclear()


def process_group(pdbid, group: pl.DataFrame, output_dir: str, bpargs: BPArgs):
    pdbid = str(pdbid)
    loaded = False
    def load_if_needed() -> None:
        nonlocal loaded
        if not loaded:
            load(None, pdbid)
            loaded = True
    pdbdir = os.path.join(output_dir, pdbid)
    os.makedirs(pdbdir, exist_ok=True)
    for chain1, nt1, ins1, alt1, chain2, nt2, ins2, alt2 in zip(group["chain1"], group["nr1"], group["ins1"], group["alt1"], group["chain2"], group["nr2"], group["ins2"], group["alt2"]):
        ins1 = None if ins1 == ' ' else ins1
        ins2 = None if ins2 == ' ' else ins2
        alt1 = None if alt1 == '?' else alt1
        alt2 = None if alt2 == '?' else alt2
        output_file = os.path.join(pdbdir, f"{chain1}_{nt1}{ins1 or ''}{alt1 or ''}-{chain2}_{nt2}{ins2 or ''}{alt2 or ''}")
        if bpargs.incremental and os.path.exists(output_file + ".png"):
            if bpargs.movie == 0 or os.path.exists(output_file + ".mp4"):
                print(f"Skipping {output_file} because it already exists")
                continue

        load_if_needed()
        make_pair_image(output_file + ".png", pdbid, chain1, nt1, ins1, alt1, chain2, nt2, ins2, alt2, bpargs)
        make_pair_rot_movie(output_file + ".mp4", pdbid, chain1, nt1, ins1, alt1, chain2, nt2, ins2, alt2, bpargs)

    cmd.reinitialize()

def make_images(df: pl.DataFrame, output_dir: str, bpargs: BPArgs):
    for pdbid, group in sorted(df.groupby("pdbid")):
        process_group(pdbid, group, output_dir, bpargs)
    # if cmd.is_gui_thread():
        # cmd.quit()

def make_images_mp(df: pl.DataFrame, output_dir: str, threads: int, bpargs: BPArgs):
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(threads) as pool:
        processes = [
            pool.apply_async(process_group, (pdbid, group, output_dir, bpargs))

            for pdbid, group in sorted(df.groupby("pdbid"))
        ]
        for p in processes:
            p.get()


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description="Generate contact images")
    parser.add_argument("input", help="Input CSV file", nargs="+")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--threads", "-t", type=int, default=1, help="Number of threads to use")
    parser.add_argument("--standard-orientation", type=bool, default=True, help="When set to true, orient the base pair such that the first nucleotide is always left and the N1/N9 - C1' is along the y-axis (N is above C)")
    parser.add_argument("--label-atoms", type=str, nargs="*", default=[], help="Atom names to label")
    parser.add_argument("--movie", type=int, default=0, help="If not zero, produce a rotating animation of the base pair")
    parser.add_argument("--incremental", type=bool, default=False, help="Generate the image/video only if it does not exist yet")
    args = parser.parse_args(argv)

    import pair_csv_parse
    df = pair_csv_parse.scan_pair_csvs(args.input).collect()
    bpargs = BPArgs(args.label_atoms, args.standard_orientation, args.movie, args.incremental)
    if args.threads == 1:
        make_images(df, args.output_dir, bpargs)
    else:
        make_images_mp(df, args.output_dir, args.threads, bpargs)

if __name__ == "__main__" or __name__ == "pymol":
    main(sys.argv[1:])


