# dualde

**Dual-graph deconvolution of enzyme fitness**

The project and Python package are **dualde** (lowercase), so the name is not
read as directed evolution (DE).

dualde separates evolutionary constraint from thermodynamic stability for
single-residue substitutions, then traces putative catalytic effects along
an [EVcouplings](https://evcouplings.org) residue–residue graph.

This repository is the accompanying code and a fully specified **NUDT15 /
PDB 5lpg** example. All paths are relative to the clone; reproducing the
example does not require a GPU or download of model weights.

---

## Overview

| Stage | Notebook | Role |
|-------|----------|------|
| Structure | `01_mapping_and_contacts.ipynb` | UniProt↔PDB map; ligand and protein contacts |
| Structure | `02_select_structures.ipynb` | CA-RMSD clustering of related PDB entries |
| Scores | `03_evolutionary_scores.ipynb` | ESM-2 650M mutation scores |
| Scores | `04_stability_spurs.ipynb` | SPURS ΔΔ*G* (raw; deconvolution uses −ΔΔ*G*) |
| Deconvolution | `05_deconvolution.ipynb` | Rank-difference KDE of evolution vs stability |
| Routing | `06_network_routing.ipynb` | Yen *k*-shortest paths on the coupling graph |

Evolutionary **ESM-2** scores are the only sequence model dualde infers.
**EVmutation** single-mutant scores and the **coupling graph** are obtained
from the [EVcouplings webserver](https://evcouplings.org) (both are shipped
for NUDT15).

Live **ESM-2 650M** and **SPURS** inference run **only on CUDA**. On CPU,
notebooks 03 and 04 load the precomputed tables.

---

## Requirements

- Python ≥ 3.10
- The NUDT15 example: ~50 MB disk; no GPU
- Optional inference (ESM-2 650M or SPURS): NVIDIA GPU, CUDA-enabled
  PyTorch, Hugging Face access for weights

Core dependencies: `pandas`, `numpy`, `scipy`, `biopython`, `networkx`,
`matplotlib`, `seaborn`.

---

## Installation

```bash
git clone https://github.com/sgb67/dualde.git
cd dualde
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[notebooks]"
```

Confirm the example tree resolves:

```bash
python -c "from dualde.paths import example_dir; print(example_dir())"
```

If `git clone https://github.com/sgb67/dualde.git` requires credentials, use
SSH (`git@github.com:sgb67/dualde.git`) after access is granted, or unpack
an archival zip.

---

## Reproducing the NUDT15 example

From the repository root, with the environment active:

```bash
jupyter lab notebooks/
```

Execute notebooks **01 through 06 in order** (Kernel → Restart & Run All).
Each notebook writes to `examples/NUDT15_HUMAN/outputs/` and inspects the
tables it produces. A later notebook that cannot find an input usually
means the previous notebook did not complete.

Expected runtime on CPU (default CSVs): a few minutes. GPU rescoring of
ESM-2 650M or SPURS is optional and is skipped automatically without CUDA.

---

## Optional GPU inference

Weights are **not** stored in this repository.

**ESM-2 650M** (notebook 03) — published [`evedesign`](https://pypi.org/project/evedesign/):

```bash
pip install -e ".[evo]"
export HF_HOME="$PWD/.cache/huggingface"
```

If `torch.cuda.is_available()` is true, notebook 03 runs
`esm2_t33_650M_UR50D` and writes `ESM2_650M`. Otherwise it loads
`scores/evolutionary.csv`.

**SPURS** (notebook 04) — official package
[luo-group/SPURS](https://github.com/luo-group/SPURS) branch
[`fix/py311-inference-beta`](https://github.com/luo-group/SPURS/tree/fix/py311-inference-beta).
Do **not** install the PyPI project named `spurs`.

```bash
pip install -e ".[spurs]"
git clone https://github.com/luo-group/SPURS.git
cd SPURS && git checkout fix/py311-inference-beta
pip install -e . --no-deps
cd ..
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
```

If CUDA is available, notebook 04 calls `get_SPURS_from_hub` +
`parse_pdb_resolved_only`. Otherwise it loads `scores/5lpg_spurs.csv`.
Cached values are raw ΔΔ*G* (wild type = 0). Deconvolution uses −SPURS so
that stabilizing substitutions rank higher.

---

## EVcouplings webserver

dualde does not fit a Potts model. For a new protein:

1. Submit the FASTA at [evcouplings.org](https://evcouplings.org).
2. Download the single-mutant / EVmutation table (labels such as `Q144A`).
3. Download `*CouplingScoresCompared_all.csv` (`i`, `j`, `probability`,
   `dist`) for notebook 06.

NUDT15 already includes `scores/evolutionary.csv` (ESM-2 plus an
`EVmutation` column) and
`couplings/b0.5_CouplingScoresCompared_all.csv`.

---

## Applying dualde to a new protein

1. **Mapping.** Provide a PDB and a UniProt map (shipped JSON schema, or
   SIFTS: download
   [pdb_chain_uniprot.csv.gz](https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz)
   and call `dualde.mapping.mapping_entry_from_sifts`). JSON `structure_*`
   fields are SEQRES indices; dualde infers the SEQRES→ATOM offset.
2. **Contacts.** `dualde.contacts.residue_contacts` on the **uncleaned** PDB
   (ligands must be present).
3. **Evolution.** ESM-2 650M on GPU, or a precomputed `mutation`/`ESM2_650M`
   table. EVmutation and couplings: EVcouplings webserver.
4. **Stability.** SPURS on GPU, or a UniProt-numbered raw ΔΔ*G* table.
5. **Deconvolution and routing.** Notebooks 05–06, with protein-specific
   activity/abundance cutoffs and ligand residue names.

---

## Repository layout

```
dualde/                  Python package
notebooks/               01–06, NUDT15 driver
examples/NUDT15_HUMAN/
  NUDT15_HUMAN.fasta
  pdb/5lpg.pdb           raw coordinates (ligand 71V + SEQRES)
  pdb/5lpg_cleaned.pdb   chain A, protein atoms
  pdb/{5bon,6t5j,7b66}*  additional entries for RMSD
  mapping/5lpg_A.json
  mapping/pdb_hits.json
  scores/evolutionary.csv
  scores/5lpg_spurs.csv
  dms/yan2020_*.csv
  couplings/b0.5_CouplingScoresCompared_all.csv
  outputs/               notebook products
```

NUDT15 ligand names used for contacts and path sources:
`MKB`, `SYW`, `4DG`, `UO8`, `RQ2`, `RTW`, `RDZ`, `71V`. PDB 5lpg contains
**71V**.

---

## Citation

Please cite the dualde manuscript when using this software. Please also cite
the methods dualde wraps:

- **ESM-2.** Lin, Z. *et al.* Language models of protein sequences at the
  scale of evolution enable accurate structure prediction. *Science*
  **379**, 1123–1130 (2023).
- **SPURS.** Li, P. & Luo, Y. Generalizable and scalable protein stability
  prediction with rewired protein generative models. *Nat. Commun.* (2025).
  doi:10.1038/s41467-025-67609-4
- **EVcouplings / EVmutation.** Hopf, T. A. *et al.* Mutation effects
  predicted from sequence co-variation. *Nat. Biotechnol.* **35**, 128–135
  (2017).
- **NUDT15 DMS.** Yan, R. *et al.* (2020), activity and abundance tables
  shipped under `examples/NUDT15_HUMAN/dms/`.

See `CITATION.cff`.

---

## License

MIT (see `LICENSE`).
