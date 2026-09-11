# DualDE

Dual-graph deconvolution of enzyme fitness: map a structure to UniProt, score
substitutions with evolutionary models and SPURS, separate catalytic from
stability effects, and route the catalytic signal over an EVcouplings graph.

This repository ships a path-agnostic **NUDT15 / 5lpg** example. Default runs
are CPU-only and use precomputed CSVs (no GPU, no Hugging Face weights).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Core dependencies: `pandas`, `numpy`, `scipy`, `biopython`, `networkx`,
`matplotlib`, `seaborn`.

### Optional extras

**SPURS (notebook 04, live scan).** Do **not** `pip install spurs` from PyPI
(that is a different package). Clone the SPURS repo, check out
`fix/py311-inference-beta`, and install DualDE extras plus the local package
without its dependency tree:

```bash
pip install -e ".[spurs]"
git clone https://github.com/luo-group/SPURS.git
cd SPURS
git checkout fix/py311-inference-beta
pip install -e . --no-deps
```

The GitHub project is [luo-group/SPURS](https://github.com/luo-group/SPURS)
([`fix/py311-inference-beta`](https://github.com/luo-group/SPURS/tree/fix/py311-inference-beta)).

Set cache locations before a live SPURS run (weights are **not** in this repo):

```bash
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
```

**Evolutionary models (notebook 03, live re-run).** Uses published `evedesign`
from PyPI for ESM2 / EVmutation2:

```bash
pip install -e ".[evo]"
```

EVmutation2 needs a **user-supplied** checkpoint path (do not commit the
`.ckpt`). ESM2 downloads weights into `HF_HOME`.

## NUDT15 / 5lpg pipeline

Notebooks live in `notebooks/` and write to
`examples/NUDT15_HUMAN/outputs/`. Run in order:

| Step | Notebook | GPU? | Default behaviour |
|------|----------|------|-------------------|
| 1 | `01_mapping_and_contacts.ipynb` | no | Load shipped `mapping/5lpg_A.json`; compute 5lpg ligand/protein contacts |
| 2 | `02_select_structures.ipynb` | no | CA-RMSD cluster of a small PDB set; example structure remains **5lpg** |
| 3 | `03_evolutionary_scores.ipynb` | optional | Load `scores/evolutionary.csv`; optional evedesign re-run |
| 4 | `04_stability_spurs.ipynb` | optional | Load cached `scores/5lpg_spurs.csv` (raw ddG), or run SPURS |
| 5 | `05_deconvolution.ipynb` | no | Rank-difference KDE of evo vs `-spurs` and vs DMS stability |
| 6 | `06_network_routing.ipynb` | no | Yen *k*-shortest paths on the shipped coupling graph |

Cached SPURS values are **raw ddG** (WT = 0). Deconvolution uses `-spurs` so
stabilizing mutations rank higher, matching `neg_spurs_multi_min` in the
source notebooks.

## Mapping JSON / SIFTS

The example mapping is a 5lpg chain A slice and runs offline. To rebuild from
SIFTS (no jackhmmer):

1. Download [pdb_chain_uniprot.csv.gz](https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz) from EBI.
2. Call `dualde.mapping.mapping_entry_from_sifts(...)` (or
   `download_sifts_table` then that helper).

`structure_*` coordinates in the JSON are SEQRES indices. DualDE infers the
SEQRES→PDB residue offset from SEQRES records in the raw PDB.

## Example layout

```
examples/NUDT15_HUMAN/
  NUDT15_HUMAN.fasta
  pdb/5lpg.pdb                 # raw (ligands + SEQRES; ligand 71V)
  pdb/5lpg_cleaned.pdb         # chain A protein only
  pdb/{5bon,6t5j,7b66}*.pdb    # extra structures for RMSD clustering
  mapping/5lpg_A.json
  mapping/pdb_hits.json        # NUDT15 PDB/chain list for RMSD
  scores/evolutionary.csv
  scores/5lpg_spurs.csv        # raw ddG cache
  dms/yan2020_activity.csv
  dms/yan2020_stability.csv
  couplings/b0.5_CouplingScoresCompared_all.csv
  aln/NUDT15_HUMAN_b0.7.a3m    # optional evo re-run only
  outputs/                     # notebook products
```

NUDT15 active-site ligand residue names used for contacts / routing:
`MKB, SYW, 4DG, UO8, RQ2, RTW, RDZ, 71V`. 5lpg itself contains **71V**
(plus protein–protein contacts); the other names appear in other NUDT15 PDBs.

## Out of scope

ThermoMPNN, RSA analog, jackhmmer multi-PDB discovery, EVE retraining, and
Boltz are not part of this first-cut example.
