# Transfer the `alignment` Conda Environment

This guide explains how to export the current conda environment named `alignment`
and recreate it on another server. After recreating the conda environment, you
should reinstall the local package from `experiments-lm-alignment` so that
`experiments-lm-alignment/src/alignment` is importable on the new server.

## 1. Export the environment on the current server

From the repository root:

```bash
cd /home/ldy/Scalable-preference-optimization-and-evaluation
conda env export -n alignment --no-builds | sed '/^prefix:/d' > alignment_environment.yml
```

The `--no-builds` option keeps package versions but removes conda build strings,
which usually makes the environment easier to recreate on another Linux server.

Optional: if the target server is very similar and you want a more exact export,
you can also create a fully pinned file:

```bash
conda env export -n alignment | sed '/^prefix:/d' > alignment_environment_full.yml
```

Use `alignment_environment.yml` first unless you specifically need the fully
pinned version.

## 2. Copy the repository and environment file to the new server

If the repository is already managed by git, the cleanest option is to clone or
pull the same code on the new server.

Example:

```bash
git clone <REPO_URL> Scalable-preference-optimization-and-evaluation
```

Then copy the exported environment file to the new server:

```bash
scp alignment_environment.yml <USER>@<SERVER>:/path/to/Scalable-preference-optimization-and-evaluation/
```

If the repository is not available through git, copy the whole folder instead:

```bash
rsync -av --progress /home/ldy/Scalable-preference-optimization-and-evaluation/ \
  <USER>@<SERVER>:/path/to/Scalable-preference-optimization-and-evaluation/
```

## 3. Create the conda environment on the new server

On the new server:

```bash
cd /path/to/Scalable-preference-optimization-and-evaluation
conda env create -f alignment_environment.yml
conda activate alignment
```

If an environment named `alignment` already exists on the new server, either
remove it first:

```bash
conda env remove -n alignment
conda env create -f alignment_environment.yml
conda activate alignment
```

or create a new environment name:

```bash
conda env create -n alignment_new -f alignment_environment.yml
conda activate alignment_new
```

## 4. Reinstall the local `alignment` package

The exported conda environment installs third-party dependencies, but local
editable packages normally need to be installed again on the new server.

Install the package from `experiments-lm-alignment`:

```bash
cd /path/to/Scalable-preference-optimization-and-evaluation/experiments-lm-alignment
pip install -e .
```

If you need development extras, use:

```bash
pip install -e ".[dev]"
```

## 5. Verify the installation

Check that Python imports the local package from this repository:

```bash
python -c "import alignment; print(alignment.__file__)"
```

The output should point to:

```text
/path/to/Scalable-preference-optimization-and-evaluation/experiments-lm-alignment/src/alignment/...
```

You can also verify the main package metadata:

```bash
python -m pip show alignment-handbook
```

## 6. Common issues

### CUDA or PyTorch mismatch

If the new server has a different GPU driver or CUDA setup, PyTorch-related
packages may need to be reinstalled for that machine. Check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If CUDA is unavailable, reinstall PyTorch using the command recommended for the
new server's CUDA version.

### Missing local package after environment creation

If this fails:

```bash
python -c "import alignment"
```

run the editable install again:

```bash
cd /path/to/Scalable-preference-optimization-and-evaluation/experiments-lm-alignment
pip install -e .
```

### Pip packages fail during `conda env create`

If the environment creation fails while installing pip packages, first create
the conda dependencies, activate the environment, and then install the local
package manually:

```bash
conda env create -f alignment_environment.yml
conda activate alignment
cd /path/to/Scalable-preference-optimization-and-evaluation/experiments-lm-alignment
pip install -e .
```

