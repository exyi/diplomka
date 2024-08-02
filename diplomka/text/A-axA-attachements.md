# Attachments {.unnumbered}

| Directory | Description |
|----|------------------------------|
| `scripts/` | Data analysis script |
| `data/`   | Contains the reference set of nucleotides ([sec. @sec:filter]) and the parameter limit table ([sec. @sec:filter]). |
| `webapp/` | Source code of the basepairs.datmos.org web application. |
| `webapp-build` | A compiled application including prepared data files. |
| `thesis/` | Source text of the thesis itself and the associated scripts. |
| `pipeline-refset.sh` | Example script to run the entire pipeline on the reference set |

The directories with code contain a brief README.md file with setup instructions.
Each Python script writes out how it is used when executed with the `--help` option.

## Dependencies and build instructions

The data processing scripts are written in Python, and we have verified that it works on versions 3.11 or 3.12.
We manage dependencies using Poetry by issuing a `poetry install` command in the `scripts/` directory.
However, the built-in `venv` command is sufficient to install the required Python libraries.

Listing: Installation of Python dependencies using the Python built-in `venv` module.

```shell
python3.12 -m venv basepairs-venv
basepairs-venv/bin/pip install -r scripts/requirements.txt

# run a script in the virtual environment:
basepairs-venv/bin/python scripts/pairs.py ...
```

The web application is a SvelteKit application with dependencies managed through `npm` (Node package manager), and is installed by the `npm install` command.
It is verified to work on Node version 22.3.
The application can be started in development mode using `npm run dev` or compiled into the static files using `npm run build`.

## Pre-built web application

The web application described in @sec:tuning-app is a static website and a limited version is part of the attachment in the `webapp-build/` directory.
The application includes pre-computed Parquet data files for the reference set.
Images of basepairs and the data files for the entire PDB are omitted to keep the attachment size reasonable.

To display the website, it is necessary to start a web server in the directory --- simply opening the index.html file will not work due to security restrictions.
The simplest option is probably to use the Python built-in module: `python -m http.server 12345`, and navigate to http://localhost:12345

## Example pipeline

The `pipeline-refset.sh` shellscript illustrates how to use the Python scripts to analyze PDB structures, assign basepairs according, compare it with FR3D, and produce the partitioned data files for the web application.

By default, it operates only on structures in the reference set (@sec:filter) to limit the computational resources required.
Except for the final part (running `pair_distributions.py`), it runs multithreaded, by default on half of the available cores.
