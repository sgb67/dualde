# dualde

**Dual-graph deconvolution of enzyme fitness**

**dualde** separates evolutionary constraint from stability for
single-residue substitutions, and then traces putative catalytic effects along
an [EVcouplings](https://evcouplings.org) residue–residue graph.

This repository is the accompanying software and a complete **NUDT15 /
PDB 5lpg** example. Paths are relative to the clone. Reproducing the example
does not require a GPU or model weights.

---

## Overview

| Stage | Notebook | Role |
|-------|----------|------|
| Structure | `01_mapping_and_contacts.ipynb` | UniProt↔PDB fragments for all homologous hits |
| Structure | `02_select_structures.ipynb` | Fetch hits ≥60% seqID; ligand/homodimer contacts; RMSD for the SPURS PDB |
| Scores | `03_evolutionary_scores.ipynb` | ESM-2 650M mutation scores |
| Scores | `04_stability_spurs.ipynb` | SPURS ΔΔ*G* (raw; deconvolution uses −ΔΔ*G*) |
| Deconvolution | `05_deconvolution.ipynb` | Rank-difference KDE of evolution vs stability |
| Routing | `06_network_routing.ipynb` | Yen *k*-shortest paths on the coupling graph |

**ESM-2 650M** is the only sequence model dualde infers. **EVmutation** scores
and the **coupling graph** come from the [EVcouplings webserver](https://evcouplings.org)
(both are included for NUDT15).

Live ESM-2 650M and SPURS inference run **only on CUDA**. Without a GPU,
notebooks 03 and 04 load the precomputed tables.

Protein-specific settings live in `examples/<protein>/config.json` (PDB id,
chain, ligand residue names, DMS cutoffs). The Python API does not hard-code
NUDT15.

---

## Requirements

- Python **3.10** (conda)
- NUDT15 example: tens of MB of disk; no GPU
- Optional live ESM-2 650M: Python ≥ 3.12, NVIDIA GPU, CUDA PyTorch
- Optional live SPURS: NVIDIA GPU and the GitHub SPURS package

Core dependencies: `pandas`, `numpy`, `scipy`, `biopython`, `networkx`,
`matplotlib`, `seaborn`.

---

## Installation

From a **fresh conda environment with Python 3.10**. Create the env **on
scratch** (`-p`), not under `$HOME`. Source `scripts/env.sh` **before**
`conda` or `pip` so package caches stay in `$DUALDE_CACHE` (default
`<clone>/.cache`). That directory must not be under `$HOME`.

```bash
git clone https://github.com/sgb67/dualde.git
cd dualde

# optional: export DUALDE_CACHE=/path/on/scratch/dualde-cache
source scripts/env.sh

conda create -y -p "$PWD/.conda" python=3.10 pip
conda activate "$PWD/.conda"

python -m pip install -U pip
python -m pip install -e ".[notebooks]"
python -m ipykernel install --sys-prefix --name dualde --display-name dualde

python -c "from dualde import load_example, cache_root; print(load_example()[0]); print(cache_root())"
```

`".[notebooks]"` installs Jupyter (`jupyter`, `ipykernel`, `nbclient`,
`nbconvert`, `nbformat`). Later shells:

```bash
cd dualde
source scripts/env.sh
conda activate "$PWD/.conda"
```

If HTTPS clone requires credentials, use SSH (`git@github.com:sgb67/dualde.git`)
or an archival zip.

---

## Reproducing the NUDT15 example

```bash
cd dualde
source scripts/env.sh
conda activate "$PWD/.conda"
jupyter lab notebooks/
```

Run notebooks **01–06 in order** (Restart & Run All). Outputs go to
`examples/NUDT15_HUMAN/outputs/`. CPU runtime with the shipped tables is a
few minutes.

---

## Optional GPU inference

Weights are not stored in this repository.

**ESM-2 650M** (notebook 03) needs **Python ≥ 3.12** (`evedesign`). Use a
separate conda env; do not add `[evo]` to the 3.10 example environment.

```bash
source scripts/env.sh
conda activate "$PWD/.conda"   # or a python=3.12 env
pip install -e ".[evo]"
```

**SPURS** (notebook 04) — [luo-group/SPURS](https://github.com/luo-group/SPURS)
branch [`fix/py311-inference-beta`](https://github.com/luo-group/SPURS/tree/fix/py311-inference-beta).
Do not `pip install spurs` from PyPI.

```bash
source scripts/env.sh
conda activate "$PWD/.conda"
pip install -e ".[spurs]"
git clone https://github.com/luo-group/SPURS.git
cd SPURS && git checkout fix/py311-inference-beta
pip install -e . --no-deps
cd ..
```

Cached SPURS values are raw ΔΔ*G* (wild type = 0). Deconvolution uses −SPURS.

---

## EVcouplings webserver

dualde does not fit a Potts model. For a new protein:

1. Submit the FASTA at [evcouplings.org](https://evcouplings.org).
2. Download the single-mutant / EVmutation table (labels such as `Q144A`).
3. Download `*CouplingScoresCompared_all.csv` (`i`, `j`, `probability`, `dist`).

---

## Applying dualde to a new protein

Copy `examples/NUDT15_HUMAN/` (or start an empty `examples/<id>/` tree) and
edit `config.json`. Then:

1. **Mapping.** A JSON record with one or more `position_mapping` fragments.
   Each fragment lists UniProt and **PDB ATOM** residue ranges (and optionally
   SEQRES). SIFTS: download
   [pdb_chain_uniprot.csv.gz](https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz)
   and call `mapping_entries_from_sifts`. Discontinuous chains are multiple
   fragments, not a single offset.
2. **Structures and contacts.** `select_structures` fetches every mapped PDB.
   Keep chains whose identity to the query is ≥ `min_identity` (default 0.6,
   over the full query length). `residue_contacts_from_structures` then scores
   ligands and homodimer interfaces on those hits; protein partners below the
   identity cutoff are ignored. RMSD clustering is only used to justify the
   representative PDB for SPURS; it does not drop contact sites.
3. **Evolution.** ESM-2 650M on GPU, or a table with `mutation` / `ESM2_650M`.
4. **Stability.** SPURS on GPU, or a UniProt-numbered raw ΔΔ*G* table.
5. **Deconvolution and routing.** Notebooks 05–06 use the cutoffs and ligands
   in `config.json`.

---

## Citation

Please cite the dualde manuscript and the methods it wraps:

- **ESM-2.** Lin, Z. *et al.* *Science* **379**, 1123–1130 (2023).
- **SPURS.** Li, P. & Luo, Y. *Nat. Commun.* (2025). doi:10.1038/s41467-025-67609-4
- **EVcouplings.** Hopf, T. A. *et al.* *Nat. Biotechnol.* **35**, 128–135 (2017).
- **NUDT15 DMS.** Yan, R. *et al.* (2020), tables under `examples/NUDT15_HUMAN/dms/`.

See `CITATION.cff`.

## License

MIT (`LICENSE`).
