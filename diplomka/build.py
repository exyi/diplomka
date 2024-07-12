#!/usr/bin/env python3

import datetime
import subprocess, os, re, sys, json, dataclasses, argparse, shutil, typing as ty
import requests
import pandocfilters as pf
import html

import tzlocal

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
    journal: ty.Optional[list[str]]
    url: ty.Optional[str] = None
    isbn: ty.Optional[str] = None
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
        authors=[["", author["name"]] if "name" in author else [author.get("given", ""), author.get("family", "???")] for author in data["author"]],
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

def assign_citation_ids(citations: dict[str, CitationInfo]) -> dict[str, CitationInfo]:
    assigned_ids = dict()
    for url, cit in citations.items():
        if cit.id is not None:
            assigned_ids[cit.id] = cit

    for url, cit in citations.items():
        if cit.id:
            preferred_id = cit.id
        elif len(cit.authors) > 3:
            preferred_id = f"{cit.authors[0][1][0]}{cit.published_date_parts[0] % 100:02d}"
        else:
            preferred_id = f"{''.join(author[1][0:1] for author in cit.authors)}{cit.published_date_parts[0] % 100:02d}"
        if preferred_id in assigned_ids and assigned_ids[preferred_id] != cit:
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
            assert isinstance(val, str)
            result.append(val)
        elif key == "Space":
            result.append(" ")
        elif key == "Code":
            assert isinstance(val[1], str)
            result.append(val[1])
    # if isinstance(pandoc_json, dict)
    #     core(pandoc_json['t'], pandoc_json.get('c'), "", {})
    pf.walk([pandoc_json], core, "", {})
    return result

def collect_links(out: dict[str, list[str]]):
    def core(key, val, fmt, meta):
        if key == "Link":
            attr, content, [url, wtf] = val
            assert not wtf, wtf
            if url.startswith("http:") or url.startswith("https:"):
                data = "".join(get_str_content(content))
                out[url] = out.get(url, []) + [ data ]

    return core

def convert_links(citations: dict[str, CitationInfo]):
    def core(key, val, fmt, meta):
        if key == "Link":
            attr, content, [url, wtf] = val
            url: str
            content_text = "".join(get_str_content(content))
            assert not wtf, wtf
            if ['data-footnote-link', 'true'] in attr[2]:
                pass
            elif url in citations:
                cit = citations[url]
                citation_ref = pf.Span(
                        ['', ["citation-ref"], [ ["data-citation-id", str(cit.id)], ["data-citation-seq", str(cit.seq)] ] ],
                        [pf.Space(),  pf.Str(f"[{cit.id}]")])
                print(f"Reference: {url} -> {content_text} [{cit.id}]")
                new_content = [ *content, citation_ref ] if content_text.lower() != url.lower() else [ citation_ref ]
                new_url = url
                new_url = f"#ref-{cit.id}"
                return pf.Link(attr, new_content,[ new_url, ''])
            elif url.startswith("http:") or url.startswith("https:"):
                print(f"External link: {url} -> {content_text} ({repr(attr)})")
                # print(f"External link: {url} -> {content_text} ({repr(content)})")
                if 'link-no-footnote' in attr[1]:
                    return # skip css class
                if url.lower() == (f"https://www.rcsb.org/structure/" + content_text).lower():
                    # skip PDB links, it is obvious
                    return
                url_html = html.escape(url)
                breakable_url = re.sub(r"(([/][#]?|&amp;|[=])+)", r"\1&#8203;", url)
                footnote = pf.Span(
                    # ['', [], []],
                    ['', ["link-footnote"], []],
                    [
                        # pf.Link( ['', [], [ ['data-footnote-link', 'true'], ['href', url]]], [pf.Str(f"{url}")], [ url, '' ]),
                        pf.RawInline('html', f"<a href='{url_html}'>{breakable_url}</a>")
                    ]
                )
                new_content = [ *content, footnote ]
                link = pf.Link([ attr[0], attr[1], attr[2] + [ ['data-footnote-link', 'true'], ['target', '_blank'], ['rel', 'noopener noreferrer'] ]], content, [url, ''])
                # return link
                return pf.Span(
                    ['', [], []],
                    # ['', ["link-footnote"], []],
                    [link, footnote]
                )
            elif (m := re.match(r"(./)?(?P<filename>[-a-z0-9]+)[.](md)#(?P<hash>.*)", url, re.IGNORECASE)) is not None:
                # internal link - remove the filename, since it gets smashed into one file
                new_url = f"#{m.group('hash')}"
                # new_url = f"{m.group('filename')}.json#{m.group('hash')}"
                return pf.Link(attr, content, [new_url, ''])
            else:
                pass
    return core

def format_author(author: list[str], shorten_first_name) -> str:
    first_name = author[0]
    if shorten_first_name:
        first_name = " ".join(n[0] + "." for n in first_name.split())
    else:
        first_name = " ".join((n if len(n) > 1 else n + ".") for n in first_name.split())
    last_name = author[1]
    return f"{first_name} {last_name}".strip()

def format_date(date_parts: list[int]) -> str:
    if len(date_parts) == 0:
        return "date???"
    if len(date_parts) == 1:
        return str(date_parts[0])
    month_names = [ "Jan.", "Feb.", "March", "April", "May", "June", "July", "Aug.", "Sep.", "Oct.", "Nov.", "Dec." ]
    return f"{month_names[date_parts[1]-1]} {date_parts[0]}"

def generate_references(cit_map: dict[str, CitationInfo]):
    cits = sorted(cit_map.items(), key=lambda c: (c[1].published_date_parts, c[1].seq))

    blocks = [
        pf.Header(1, ["references", [], []], [pf.Str("References")]),
    ]
    for id, cit in cits:
        links = []
        link_attr = 'target=_blank rel="noopener noreferrer"'
        if cit.doi:
            links.append(f'<span class="references-link references-doi"><a {link_attr} href="https://doi.org/{html.escape(cit.doi)}">DOI: <span class="references-doi-body">{html.escape(cit.doi)}</span></a></span>')
        if cit.url and cit.url != cit.doi:
            links.append(f'''
                <span class="references-link references-url">
                    <a {link_attr} href="{html.escape(cit.url)}">Available at: <span class="references-url-body">{html.escape(cit.url)}</span></a>
                    <br>(<a {link_attr} href="https://web.archive.org/web/20240712120000/{html.escape(cit.url)}">web.archive.org</a>)
                </span>''')
            
        if cit.isbn:
            links.append(f'<span class="references-link references-isbn"><a {link_attr} href="https://isbnsearch.org/isbn/{html.escape(cit.isbn)}">ISBN: <span class="references-isbn-body">{html.escape(cit.isbn)}</span></a></span>')
        if cit.doi:
            links.append(f'<span class="references-link references-doi"><a {link_attr} href="https://sci-hub.se/{html.escape(cit.doi)}">sci-hub: <span class="references-doi-body">{html.escape(cit.doi)}</span></a></span>')
        authors = cit.authors
        if len(authors) > 20 or ["", "???"] in authors:
            authors = [a for a in authors if a[1] != "???"]
            authors = authors[:5] + [["", "et al."]]
        authors = ", ".join(format_author(a, shorten_first_name=len(authors) > 3) for a in authors)
        if cit.journal and len(cit.journal) > 0:
            j = cit.journal if isinstance(cit.journal, str) else cit.journal[0]
            journal = f'<span class="references-journal">{html.escape(re.sub(r"(\s|[,])+$", "", j))}</span>'
        else:
            journal = ""
        body = f"""
            <span class="references-authors">{html.escape(authors)}</span>
            <span class="references-date">{format_date(cit.published_date_parts)}</span>
            <span class="references-title">{cit.title}</span>
            {journal}
            {"\n".join(links)}
        """
        row = f"""
            <div class="references-table-row" id="ref-{html.escape(id)}">
                <div class="references-table-id">[{html.escape(id)}]</div>
                <div class="references-table-body">
                    {body}
                </div>
            </div>
            """
        blocks.append(pf.RawBlock("html", row))
    return {"pandoc-api-version": [1, 23, 1], "meta": {}, "blocks": blocks}

def alter_file(file, actions, format=""):
    with open(file) as f:
        altered = pf.applyJSONFilters(actions, f.read(), format)
    with open(file, "w") as f:
        f.write(altered)

def process_links(files: list[str], out_dir):
    links: dict[str, list[str]] = dict()
    phase1 = [
        collect_links(links)
    ]
    for file in files:
        with open(file) as f:
            pf.applyJSONFilters(phase1, f.read(), "")
    
    citations = get_citations_somehow(list(links.keys()))
    cit_map = assign_citation_ids(citations)
    for _, cit in sorted({ cit.seq: cit for l in links.items() if (cit:=citations.get(l[0])) is not None}.items()):
        print(f" {(cit.seq+1 if cit.seq is not None else -4086200234935): 3d} {cit.id:5s}: {cit.title} ({cit.doi or cit.url})")

    phase2 = [convert_links(citations)]
    for file in files:
        alter_file(file, phase2)

    references = generate_references(cit_map)
    with open(os.path.join(out_dir, "references.json"), "w") as f:
        json.dump(references, f, indent=4)
    files.append(os.path.join(out_dir, "references.json"))

def sentence_spacing(key: str, val: ty.Any, fmt, meta):
    if val and not isinstance(val, (str)):
        # print(key, val)
        if key == 'Para':
            new_val = [*val]
            content = new_val
        elif key == "Image":
            content = [*val[1]]
            new_val = [*val]
            new_val[1] = content
        else:
            return None
        str_c = ""
        for i in range(len(content)):
            if i > 0 and (content[i]['t'] == 'SoftBreak'):
                # print(str_c)
                if str_c.strip().endswith("."):
                    content[i] = { 't': 'Str', 'c': '\u2005 ' }

            str_c += re.sub(r"\[\w+\]", "", "".join(get_str_content(content[i])))

        return { 't': key, 'c': new_val }

def process_typography(files: list[str]):
    actions = [
        sentence_spacing
    ]
    
    for file in files:
        alter_file(file, actions)

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
    compress = ["-dDownsampleColorImages", "-dDownsampleGrayImages", "-dDownsampleMonoImages", "-dPDFSETTINGS=/screen"]
    run("Convert to PDF/A with GhostScript", "gs",
        "-dPDFA=2", "-dBATCH", "-dNOPAUSE", "-dNOOUTERSAVE",
        "-dPrinted=false", "-dPreserveAnnots=true", # don't remove links
        "-dColorImageDownsampleType=/Bicubic",
        # "-dFastWebView=true", # first page is loaded faster maybe?
        # "-sColorConversionStrategy=UseDeviceIndependentColor",
        "-sColorConversionStrategy=RGB",
        "-sProcessColorModel=DeviceRGB",
        "-sDEVICE=pdfwrite",
        "-dPDFACompatibilityPolicy=2", # error when incompatible
        # *compress,
        f"-sOutputFile={outfile}",
        "./pdf/PDFA_def.ps",
        infile, capture_output=True)
    validate_pdfa(outfile)

    recompress_pdf(outfile, os.path.join(os.path.dirname(outfile), "recompressed.pdf"))
    print("Optimized pdf:", os.path.join(os.path.dirname(outfile), "recompressed.pdf"))
    validate_pdfa(os.path.join(os.path.dirname(outfile), "recompressed.pdf"))
    return True

def recompress_pdf(infile, outfile):
    run("Recompress PDF with qpdf",
        "qpdf", infile, "--linearize", "--object-streams=generate", "--stream-data=compress", "--recompress-flate", "--compression-level=9",
        "--normalize-content=y", "--remove-unreferenced-resources=yes",
        outfile,
        capture_output=True)

def validate_pdfa(file):
    try:
        r = run("PDF/A Verification", "verapdf", "--profile", "./pdf/UK_validation_profile.xml", file, capture_output=True, check=False)
    except FileNotFoundError:
        eprint("verapdf not found, skipping PDF/A validation")
        return False

    if r.returncode != 0:
        eprint("PDF/A validation failed")
        eprint(r.stdout)
        return False
    assert r.stdout is not None
    if 'nonCompliant="0"' in r.stdout and 'compliant="1"' in r.stdout:
        print("PDF/A validation passed")
        return True
    else:
        eprint("PDF/A validation is sus")
        return False

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
        if file.endswith('_'):
            continue

        cmd = ["pandoc", "--to=json", "--wrap=preserve", "--output=" + os.path.join(output_dir, file + ".json"), "--strip-comments"]
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

def reexport_pandoc_for_review(files, output_file):
    """
    Single markdown file for AI review using llamacpp-review.mjs
    """
    run(f"Pandoc reexport for review", "pandoc", "--to=markdown", "--wrap=preserve", "--output=" + output_file, *files)

def reexport_docx(files, out_dir, output_file):
    """
    """
    args = [
        "--data-dir=./"+out_dir,
        "--standalone",
        "--resource-path=text:html:images",
        "--metadata-file=text/metadata.yaml",
        "-F", "pandoc-crossref",
        "--toc", "--toc-depth=4",
        "--number-sections"
    ]
    # run(f"Pandoc export docx", "pandoc", *args, "--to=docx", "--output=" + output_file + ".docx", *files)
    # run(f"Pandoc export odt", "pandoc", *args, "--to=odt", "--output=" + output_file + ".odt", *files)

def pandoc_render(files, output_file):
    commit = run(f"git current commit", "git", "rev-parse", "--short", "HEAD", check=False, capture_output=True).stdout.strip()
    commit_num = run(f"git commit number", "git", "rev-list", "--count", "HEAD", check=False, capture_output=True).stdout.strip()
    cmd = ["pandoc",
           "--from=json",
           "-F", "pandoc-crossref",
        #    "--embed-resources",
           "--resource-path=text:html:images",
           "--metadata-file=text/metadata.yaml",
           f"--metadata=generated_at:{str(datetime.datetime.now(tzlocal.get_localzone()))}",
           f"--metadata=generated_from_commit:{commit}",
           f"--metadata=generated_from_commit_num:{commit_num}",
           "--template=html/template.html",
           "--standalone",
        #    "--shift-heading-level-by=1",
           "--lua-filter=html/shift-headings.lua",
           "--toc", "--toc-depth=4",
           "--number-sections", "--section-divs"]
    run(f"Render {output_file} with Pandoc", *cmd, "--to=html", "--katex", "--output=" + output_file + ".html", *files)
    # run(f"Render {output_file}.epub with Pandoc", *cmd, "--to=epub", "--output=" + output_file + ".epub", "--webtex", *files)

def main(argv):
    parser = argparse.ArgumentParser(description='Build the thesis')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--skip-pdf', action='store_true', help='Verbose output')
    parser.add_argument('--pdf', type=str, help="Input PDF file, only perform PDF/A conversion")
    args = parser.parse_args(argv)
    options.verbose = args.verbose
    options.skip_pdf = args.skip_pdf

    if args.pdf:
        convert_pdfa(args.pdf, "out/thesis-pdfa.pdf")
        print("PDF/A conversion done, see out/thesis-pdfa.pdf")
        return

    files = pandoc_parse("text", "out/parsed")
    reexport_pandoc_for_review(files, "out/for_review.md")
    process_links(files, "out/parsed")
    process_typography(files)
    reexport_docx(files, "out", "out/thesis")
    pandoc_render(files, "out/thesis")

    if not options.skip_pdf:
        build_pdf("out/thesis.html")

if __name__ == '__main__':
    main(sys.argv[1:])
