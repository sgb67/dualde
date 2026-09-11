"""dualde: dual-graph deconvolution of enzyme fitness.

Separates evolutionary constraint from thermodynamic stability for
single-residue substitutions, then routes putative catalytic signal over an
EVcouplings residue graph.
"""

from dualde.paths import example_dir, repo_root

__version__ = "0.1.0"

__all__ = ["__version__", "example_dir", "repo_root"]
