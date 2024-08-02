# Attachments {.unnumbered}

| Directory | Description |
|----|------------------------------|
| `scripts/`{.nowrap} | Data analysis script |
| `data/`{.nowrap} | Contains the reference set of nucleotides ([sec. @sec:filter]) and the parameter limit table. |
| `webapp/`{.nowrap} | Source code of the basepairs.datmos.org web application. |
| `webapp-build`{.nowrap} | A compiled application including prepared data files. |
| `thesis/`{.nowrap} | Source text of the thesis itself and the associated scripts. |
| `pipeline-refset.sh`{.nowrap} | Example script to run the entire pipeline on the reference set |

The directories with code contain a brief README.md file with setup instructions.
Each Python script writes out how it is used when executed with the `--help` option.

The source code is also available online at GitHub: <https://github.com/exyi/diplomka>.
Complete pre-computed data files partitioned by basepairing class are available online at <https://basepairs.datmos.org/tables> (about 10 GiB in total).


## Dependencies and build instructions{.unnumbered}

The data processing scripts are written in Python, and we have verified that it works on versions 3.11 or 3.12.
We manage dependencies using Poetry by issuing a `poetry install` command in the `scripts/` directory.
However, the built-in `venv` command is sufficient to install the required Python libraries.

Listing: Installation of Python dependencies using the Python built-in `venv` module. {#lst:python-venv-installdeps}

```shell
python3.12 -m venv basepairs-venv
basepairs-venv/bin/pip install -r scripts/requirements.txt

# run a script in the virtual environment:
basepairs-venv/bin/python scripts/pairs.py ...
```

The web application (@sec:tuning-app) is a SvelteKit application with dependencies managed through `npm` (Node package manager), which installs them by the `npm install` command.
We know that it works on Node versions 20.15 and 22.3.
The application can be started in development mode using the `npm run dev` command or compiled into the static files using `npm run build`.

## Pre-built web application{.unnumbered}

The web application (@sec:tuning-app) is a static website and a limited version of it is attached in the `webapp-build/` directory.
The application includes pre-computed Parquet data files for the reference set, while the rendered basepair images and the data files for the entire PDB are omitted to keep the attachment size reasonable.

To display the website, it is necessary to start a web server in the directory --- simply opening the index.html file will not work due to browser security restrictions.
The simplest option is probably to use the one built in the Python standard library: `python -m http.server 12345`, and navigate to http://localhost:12345.

The website will attempt to load the basepair images over the network from basepairs.datmos.org, and it also allows to visualize any pair using the [Mol*](https://doi.org/10.1093/nar/gkab314) viewer in the modal dialog shown after clicking on it.

## Example pipeline{.unnumbered}

The `pipeline-refset.sh` shellscript illustrates how to use the Python scripts to analyze PDB structures, assign basepairs according to the parameter limits, compare it with the FR3D assignment, and produce the partitioned data files for the web application.

By default, the pipeline operates only on PDB structures in the reference set (@sec:filter) to limit the computational resources required.
Most of the time, it runs multithreaded, by default on half of the available CPU cores.
Note that each thread requires about a gigabyte of memory.

