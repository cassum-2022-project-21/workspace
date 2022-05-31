`data_io.py` copied from [MovingPlanetsAround/ABIE](https://github.com/MovingPlanetsAround/ABIE) (which is GPLv3 licensed... ask about this)

conda evironment written to `environment.yml` (using openblas for now because AMD CPU). pip requirements written to `requirements.txt`. To load the environment use

```
{mamba,conda} env create -f environment.yml
conda activate cassum-2022-project-21
pip install -U -r requirements.txt --no-deps --no-build-isolation
```

To update the environment from the file use 

```
{mamba,conda} env update --file environment.yml
pip install -U -r requirements.txt --no-deps --no-build-isolation
```

Ideally, figures are more or less reproducible from the code in this repository with the git tag corresponding to their file name. Not sure if the simulation code is deterministic?

Steps to reproduce simulations:
- First install packages as above
- `cd sim/`
- `python ic_generator.py`
- `simon start`
- check simulation progress with `simon` 
