import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList


def get_neighborlist(atoms: Atoms, cutoff: float) -> NeighborList:
    nl = NeighborList(
        len(atoms) * [cutoff / 2],
        skin=0.0,
        # sorted=True,
        self_interaction=False,
        bothways=True,
    )
    nl.update(atoms)
    return nl


def get_neighbors_displacements(
    atoms: Atoms, nl: NeighborList, i: int
) -> tuple[np.ndarray, np.ndarray]:
    j, off = nl.get_neighbors(i)
    r = (
        atoms.positions[j]
        - atoms.positions[i]
        + (off[..., None] * atoms.cell).sum(axis=1)
    )
    return j, r
