#!/usr/bin/env python3

import subprocess, os, re, sys, json, dataclasses, argparse, shutil, typing as ty

@dataclasses.dataclass
class Options:
    verbose: bool = False

options = Options()

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

@dataclasses.dataclass
class RunResult:
    returncode: int
    process: subprocess.Popen
    stdout: ty.Optional[str]
    stderr: ty.Optional[str]

def format_cmd(cmd: ty.Sequence[str]):
    return " ".join(arg if re.match(r'[-_/.a-z0-9]+', arg, re.IGNORECASE) else repr(arg) for arg in cmd)

def run(name, *cmd, check=True, justexit=True, capture_output=False, allow_not_found=False):
    if options.verbose:
        eprint(f"{name}: {format_cmd(cmd)}")
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE if capture_output else None, stderr=subprocess.PIPE if capture_output else None)
    except FileNotFoundError as e:
        if allow_not_found:
            return RunResult(-1, False, None, None, None) #type: ignore
        else:
            raise e
    out, err = None, None
    if capture_output:
        out, err = p.communicate()
        out = out.decode("utf-8")
        err = err.decode("utf-8")
        if options.verbose:
            eprint(f"{name} stdout:")
            eprint(out)
            eprint(f"{name} stderr:")
            eprint(err)

    p.wait()
    if check and p.returncode != 0:
        eprint(f"{name} failed [{cmd[0]}: {p.returncode}]")
        eprint(f"cmd: {format_cmd(cmd)}")
        if capture_output:
            eprint("stdout:")
            eprint(out)
            eprint("stderr:")
            eprint(err)
        if justexit:
            exit(13)
        else:
            raise Exception(f"{name} failed [{p.returncode}]")
    return RunResult(p.returncode, p, out, err)

def start_server(directory, port):
    p = subprocess.Popen([ "python3", "-m", "http.server", port, "--bind", "127.0.0.1" ], cwd=directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p

def convert_pdfa(infile, outfile):
    run("Convert to PDF/A with GhostScript", "gs", "-dPDFA", "-dBATCH", "-dNOPAUSE", "-sColorConversionStrategy=UseDeviceIndependentColor", "-sDEVICE=pdfwrite", "-dPDFACompatibilityPolicy=2", f"-sOutputFile={outfile}", infile, capture_output=True)
    validate_pdfa(outfile)

def validate_pdfa(file):
    try:
        r = run("PDF/A Verification", "verapdf", file, capture_output=True, check=False)
    except FileNotFoundError:
        eprint("verapdf not found, skipping PDF/A validation")
        return False

    if r.returncode != 0:
        eprint("PDF/A validation failed")
        eprint(r.stdout)
        exit(1)
    assert r.stdout is not None
    if 'nonCompliant="0"' in r.stdout and 'compliant="1"' in r.stdout:
        print("PDF/A validation passed")
    else:
        eprint("PDF/A validation is sus")

def build_pdf(infile, outfile="out/thesis-{}.pdf", httpserver=True):
    os.makedirs("out", exist_ok=True)
    if httpserver:
        server = start_server(".", "12388")
        inurl = f'http://localhost:12388/{infile}'
    else:
        inurl = infile
        server = None
    run("Typeset PDF with Chromium", "chromium", "--headless", "--disable-gpu", "--no-pdf-header-footer", "--generate-pdf-document-outline", "--virtual-time-budget=100000", "--run-all-compositor-stages-before-draw", "--print-to-pdf=" + outfile.format('chr'), inurl)
    if server is not None:
        server.terminate()
    try:
        pdfa = convert_pdfa(outfile.format('chr'), outfile.format('pdfa'))
    except Exception as e:
        eprint("PDF/A conversion failed")
        eprint(e)
        pdfa = False
    if pdfa:
        return outfile.format('pdfa')
    else:
        return outfile.format('chr')
    

def human_sort_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def pandoc_parse(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    out_files = []
    for file in sorted(os.listdir(input_dir), key=human_sort_keys):
        cmd = ["pandoc", "--to=json", "--output=" + os.path.join(output_dir, file + ".json")]
        format = None

        if file in ['metadata.yaml']:
            continue
        elif file.endswith(".md"):
            format = "markdown+smart+gfm_auto_identifiers+superscript+subscript+inline_code_attributes+link_attributes+emoji"
        elif file.endswith(".html"):
            format = "html"
        else:
            raise Exception(f"Unknown file extension: {file}")
        
        cmd.append("--from=" + format)
        cmd.append(os.path.join(input_dir, file))
        run(f"Parse {file} with Pandoc", *cmd)
        out_files.append(os.path.join(output_dir, file + ".json"))
    return out_files

def pandoc_render(files, output_file):
    cmd = ["pandoc",
           "--from=json", "--to=html",
        #    "--embed-resources",
           "--resource-path=text:html:images",
           "--output=" + output_file,
           "--metadata-file=text/metadata.yaml",
           "--template=html/template.html",
           "--standalone",
        #    "--shift-heading-level-by=1",
           "--lua-filter=html/shift-headings.lua",
           "--toc", "--toc-depth=4",
           "--number-sections", "--section-divs", "--katex"]
    for file in files:
        cmd.append(file)
    run(f"Render {output_file} with Pandoc", *cmd)

def main(argv):
    parser = argparse.ArgumentParser(description='Build the thesis')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args(argv)
    options.verbose = args.verbose

    files = pandoc_parse("text", "out/parsed")
    pandoc_render(files, "out/thesis.html")

    build_pdf("out/thesis.html")

if __name__ == '__main__':
    main(sys.argv[1:])
