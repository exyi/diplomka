#!/usr/bin/env python3

import subprocess, os, re, sys, json, dataclasses, argparse, shutil, typing as ty
import requests
import pandocfilters

@dataclasses.dataclass
class Options:
    verbose: bool = False
    skip_pdf: bool = False

options = Options()

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    


@dataclasses.dataclass
class CitationInfo:
    doi: ty.Optional[str]
    published_date_parts: list[int]
    title: str
    authors: list[list[str]]
    journal: ty.Optional[str]
    url: ty.Optional[str]
    id: ty.Optional[str] = None
    seq: ty.Optional[int] = None
def get_doi_from_url(url: str) -> ty.Optional[str]:
    m = re.match(r"https?://doi.org/(.+)", url)
    if m is not None:
        return m.group(1)
    else:
        return None
def download_cit_info(doi: str) -> CitationInfo:
    response = requests.get(f"https://api.crossref.org/works/{doi}")
    response.raise_for_status()
    data = response.json()["message"]
    return CitationInfo(
        doi=doi,
        published_date_parts=list(map(int, (data.get("published") or data["published-print"])["date-parts"][0])),
        title=data["title"][0],
        authors=[[author["given"], author["family"]] for author in data["author"]],
        journal=data.get("container-title"),
        url=None
    )

def load_citations(file):
    map = dict()
    with open(file) as f:
        data = json.load(f)
        for id in data:
            map[id] = CitationInfo(**data[id])
    return map

def get_citations_somehow(url_list: list[str]) -> dict[str, CitationInfo]:
    auto = load_citations("cit-auto.json")
    manual = load_citations("cit.json")

    missing_urls = [url for url in url_list if url not in auto and url not in manual]
    missing_dois = [doi for url in missing_urls if (doi := get_doi_from_url(url)) is not None]
    if missing_dois:
        for url in missing_urls:
            doi = get_doi_from_url(url)
            if doi:
                auto[url] = download_cit_info(doi)
        write_new = { k: dataclasses.asdict(v) for k, v in sorted(auto.items()) }
        try:
            with open("cit-auto.json.new", "w") as f:
                json.dump(write_new, f, indent=4)
            os.rename("cit-auto.json.new", "cit-auto.json")
        finally:
            if os.path.exists("cit-auto.json.new"):
                os.remove("cit-auto.json.new")
    return {**auto, **manual}

def assign_citation_ids(citations: dict[str, CitationInfo]):
    assigned_ids = dict()
    for url, cit in citations.items():
        if cit.id is not None:
            assigned_ids[cit.id] = cit

    for url, cit in citations.items():
        if len(cit.authors) > 3:
            preferred_id = f"{cit.authors[0][1][0]}{cit.published_date_parts[0] % 100:02d}"
        else:
            preferred_id = f"{''.join(author[1][0] for author in cit.authors)}{cit.published_date_parts[0] % 100:02d}"
        if preferred_id in assigned_ids:
            for i in range(ord('b'), ord('z')+1):
                if f'{preferred_id}{chr(i)}' not in assigned_ids:
                    preferred_id = f'{preferred_id}{chr(i)}'
                    break
                if i == 'z':
                    assert False, f"cannot assign any: {preferred_id} | {list(assigned_ids.keys())}"
        cit.id = preferred_id
        assigned_ids[preferred_id] = cit

    ss = sorted(citations.values(), key=lambda c: tuple(c.published_date_parts + [ 3000, 3000, 3000 ])[0:3])
    for i, s in enumerate(ss):
        s.seq = i
    return assigned_ids

def get_str_content(pandoc_json: dict) -> list[str]:
    result = []
    def core(key, val, fmt, meta):
        if key == "Str":
            result.append(val)
    pandocfilters.walk(pandoc_json, core, "", {})
    return result

def collect_links(out: dict[str, list[str]]):
    def core(key, val, fmt, meta):
        if key == "Link":
            attr, content, [url, wtf] = val
            assert not wtf, wtf
            if url.startswith("http:") or url.startswith("https:"):
                data = " ".join(get_str_content(content))
                out[url] = out.get(url, []) + [ data ]

    return core

def convert_links(citations: dict[str, CitationInfo]):
    def core(key, val, fmt, meta):
        if key == "Link":
            attr, content, [url, wtf] = val
            assert not wtf, wtf
            if ['data-footnote-link', 'true'] in attr[2]:
                pass
            elif url in citations:
                new_content = [
                    *content,
                    pandocfilters.Span(
                        ['', ["citation-ref"], [ ["data-citation-id", str(citations[url].id)], ["data-citation-seq", str(citations[url].seq)] ] ],
                        [pandocfilters.Space(),  pandocfilters.Str(f"[{citations[url].id}]")]),
                ]
                return pandocfilters.Link(attr, new_content,[ url, ''])
            elif url.startswith("http:") or url.startswith("https:"):
                footnote = pandocfilters.Span(
                    # ['', [], []],
                    ['', ["link-footnote"], []],
                    [
                        # pandocfilters.Link( ['', [], [ ['data-footnote-link', 'true'], ['href', url]]], [pandocfilters.Str(f"{url}")], [ url, '' ]),
                        pandocfilters.RawInline('html', f"<a href='{url}'>{url}</a>")
                    ]
                )
                new_content = [ *content, footnote ]
                link = pandocfilters.Link([ attr[0], attr[1], attr[2] + [ ['data-footnote-link', 'true'] ]], content, [url, ''])
                # return link
                return pandocfilters.Span(
                    ['', [], []],
                    # ['', ["link-footnote"], []],
                    [link, footnote]
                )
            elif (m := re.match(r"(./)?(?P<filename>[-a-z0-9]+)[.](md)#(?P<hash>.*)", url, re.IGNORECASE)) is not None:
                # internal link - remove the filename, since it gets smashed into one file
                new_url = f"#{m.group('hash')}"
                # new_url = f"{m.group('filename')}.json#{m.group('hash')}"
                return pandocfilters.Link(attr, content, [new_url, ''])
            else:
                pass
    return core


def process_links(files: list[str]):
    links: dict[str, list[str]] = dict()
    phase1 = [
        collect_links(links)
    ]
    for file in files:
        with open(file) as f:
            pandocfilters.applyJSONFilters(phase1, f.read(), "")
    
    citations = get_citations_somehow(list(links.keys()))
    cit_map = assign_citation_ids(citations)
    for l in links.items():
        cit = citations.get(l[0])
        if cit:
            print(f" {cit.seq+1: 3d} {cit.id:5s}: {cit.title} ({cit.doi})")

    phase2 = [convert_links(citations)]
    for file in files:
        with open(file) as f:
            altered = pandocfilters.applyJSONFilters(phase2, f.read(), "")
        with open(file, "w") as f:
            f.write(altered)

    # TODO: generate bibliography

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
           "-F", "pandoc-crossref",
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
    parser.add_argument('--skip-pdf', action='store_true', help='Verbose output')
    args = parser.parse_args(argv)
    options.verbose = args.verbose
    options.skip_pdf = args.skip_pdf

    files = pandoc_parse("text", "out/parsed")
    process_links(files)
    pandoc_render(files, "out/thesis.html")

    if not options.skip_pdf:
        build_pdf("out/thesis.html")

if __name__ == '__main__':
    main(sys.argv[1:])
