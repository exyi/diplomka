# Attachments {.unnumbered}

| Directory | Description |
|----|------------------------------|
| scripts | Data analysis script |
| webapp | Source code of the basepairs.datmos.org web application |
| thesis | Source text of the thesis itself and associated scripts |

Each of these directories contains a brief README file with build instructions.
Each data processing script will write out its usage when executed with the `--help` option.

## Dependencies and build instructions

The data processing scripts are written in Python and we have verified that it works on versions 3.11 or 3.12.
We manage dependencies using Poetry, but the `venv` command is sufficient to install the required Python libraries.

TODO script

The web application is a SvelteKit application with dependencies managed through `npm` (Node package manager), and is installed by the `npm install` command.
It is verified to work on Node version 22.3.
The application can be started in development mode using `npm run dev` or compiled into the static files using `npm run build`.
