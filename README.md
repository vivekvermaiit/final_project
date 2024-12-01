# COMP550 Final Project Code

This repository contains code for our final project. 

"occupation-stats.tsv" has been taken from https://github.com/rudinger/winogender-schemas/blob/master/data/occupations-stats.tsv

### Conda 

Conda uses the provided `environment.yml` file.
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) installed on your system.
Once installed, open up your terminal (or Anaconda prompt if you're on Windows).
Install the environment from the specified environment file:

    conda env create --file environment.yml
    conda activate comp550-final-project

After you install, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=comp550-final-project

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you make updates to your conda `environment.yml`, you can use the update command to update your existing environment rather than creating a new one:

    conda env update --file environment.yml    

You can create a new environment file using the `create` command:

    conda env export > environment.yml

