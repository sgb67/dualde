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

| Notebook | Role |
|----------|------|
| `01_structures.ipynb` | Mapping, identity filter, contacts, RMSD |
| `02_deconvolution.ipynb` | ESM-2, EVmutation, SPURS, DMS, rank-difference KDE |
| `03_routing.ipynb` | Coupling paths and contact / RSA null tests |

Scores and the coupling graph are the tables shipped under `examples/`. Notebooks run on CPU. RSA uses [FreeSASA](https://freesasa.github.io/).

Protein-specific settings live in `examples/<protein>/config.json` (PDB id,
chain, ligand residue names, DMS cutoffs). The Python API does not hard-code
NUDT15.

---

## Requirements

- A conda environment with **Python 3.10** (already created)
- NUDT15 example: tens of MB of disk
- FreeSASA, for the RSA nulls in `03_routing.ipynb`

Core dependencies: `pandas`, `numpy`, `scipy`, `biopython`, `freesasa`,
`networkx`, `matplotlib`, `seaborn`.

---

## Installation

Activate your existing Python 3.10 conda env, then install dualde into it.

Source `scripts/env.sh` **before pip** so wheels and model caches go to
`$DUALDE_CACHE` (default `<clone>/.cache`), not `$HOME`.

```bash
cd dualde
source scripts/env.sh

python -m pip install -U pip
python -m pip install -e ".[notebooks]"
python -m ipykernel install --sys-prefix --name dualde --display-name dualde

python -c "from dualde import load_example, cache_root; print(load_example()[0]); print(cache_root())"
```

`".[notebooks]"` installs Jupyter (`jupyter`, `ipykernel`, `nbclient`,
`nbconvert`, `nbformat`).

If HTTPS clone requires credentials, use SSH (`git@github.com:sgb67/dualde.git`)
or an archival zip.

---

## Reproducing the NUDT15 example

```bash
cd dualde
# conda activate /path/to/your/<env>
source scripts/env.sh
jupyter lab notebooks/
```

Run `01_structures.ipynb`, `02_deconvolution.ipynb`, then `03_routing.ipynb`.
Outputs go to `examples/NUDT15_HUMAN/outputs/`.

Cached SPURS values are raw ΔΔ*G* (wild type = 0). Deconvolution uses −SPURS.

---

## EVcouplings webserver

dualde does not fit a Potts model. For a new protein:

1. Submit the FASTA at [evcouplings.org](https://evcouplings.org).
2. Download the single-mutant / EVmutation table (labels such as `Q144A`).
3. Download `*CouplingScoresCompared_all.csv` (`i`, `j`, `probability`, `dist`).
   `dist` is the minimum heavy-atom distance (Å), the smaller of `dist_intra`
   and `dist_multimer`. `dist_cutoff` (5 Å in the example) zeroes the coupling
   probability of pairs farther than that. It is not the Cα contact graph.

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
   ligands and homodimer interfaces on those hits
   (`interface_distance_cutoff`, all-atom). Protein partners must also
   belong to the PDBe preferred biological assembly (with the query chain);
   crystal-packing copies are ignored. RMSD clustering is only used to justify
   the representative PDB for SPURS; it does not drop contact sites.
3. **Evolution.** A table with `mutation` / `ESM2_650M`, plus `EVmutation` from EVcouplings.
4. **Stability.** A UniProt-numbered raw ΔΔ*G* table (`spurs`). Deconvolution uses −SPURS.
5. **Deconvolution, routing, and null tests.** `02_deconvolution.ipynb` writes
   the rank-difference scores, including `score_col`
   (`EVmutation_neg_spurs_model` in the example). Routing uses that column to
   choose target positions. `contact_distance_cutoff` is the Cα–Cα distance
   used only to build the contact graph for the structural null tests.
   `interface_distance_cutoff` is the separate all-atom cutoff for recording
   ligand and homodimer contacts. `functional_dist_cutoff` then marks
   functional residues on that contact table.

---

## Citation

Please cite the dualde manuscript and the methods it wraps:

- **ESM-2.** Lin, Z. *et al.* *Science* **379**, 1123–1130 (2023).
- **SPURS.** Li, P. & Luo, Y. *Nat. Commun.* (2025). doi:10.1038/s41467-025-67609-4
- **EVcouplings.** Hopf, T. A. *et al.* *Nat. Biotechnol.* **35**, 128–135 (2017). Software: Hopf, T. A. *et al.* *Bioinformatics* **35**, 1582–1584 (2019).
- **NUDT15 DMS.** Yan, R. *et al.* (2020), tables under `examples/NUDT15_HUMAN/dms/`.

See `CITATION.cff`.

## License

MIT (`LICENSE`).
