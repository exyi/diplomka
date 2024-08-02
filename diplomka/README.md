## Build instructions

Install pandoc, python-pandocfilters, ghostscript and qpdf packages

Then, for PDF output do the following:


1. run `./build.py --pagedjs`,
2. run `python -m http.server 12345`,
3. open in a Chromium-based browser: <http://localhost:12345/out/thesis.html>,
4. print to PDF,
5. \[optionally\], run `./build.py --pdf out/thesis.pdf` for post-processing.

For .epub, .docx, or .odt runt the `./build.py` script with a `--epub`, `--docx`, or `--odt` flag.


## Contents

Most figures are in the SVG format in the `img/` directory. The `pokusy/` directory contains Jupyter notebooks which were used to generate the figures. The `text/` directory contains the text split into chapters (numbers in the file names are not necessarily the same as in the final thesis).
The `html/` directory contains the CSS styles and the HTML template using Paged.js.
