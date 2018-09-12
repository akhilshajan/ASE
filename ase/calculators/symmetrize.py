"""
Provides SymmetrizedCalculator class to wrap a Calculator and preserve
spagegroup symmetry during optimisation.
"""
from __future__ import print_function
import sys
import numpy as np
from ase.calculators.calculator import Calculator
from ase.constraints import (voigt_6_to_full_3x3_stress,
                             full_3x3_to_voigt_6_stress)

# check if we have access to get_spacegroup from spglib
# https://atztogo.github.io/spglib/
has_spglib = False
try:
    import spglib                   # For version 1.9 or later
    has_spglib = True
except ImportError:
    try:
        from pyspglib import spglib  # For versions 1.8.x or before
        has_spglib = True
    except ImportError:
        pass

__all__ = ['refine', 'check', 'SymmetrizedCalculator']

def refine(at, symprec=0.01):
    """
    Refine symmetry on `at` (ase.Atoms) to precision `symprec` (float)

    Returns resulting symmetry dataset, from `spglib.get_symmetry_dataset()`.
    """
    # test orig config with desired tol
    dataset = spglib.get_symmetry_dataset(at, symprec=symprec)
    print("ase.calculators.symmetrize.refine_symmetry: loose ({}) "
          "initial symmetry group number {}, "
          "international (Hermann-Mauguin) {} Hall {}".
          format(symprec, dataset["number"],
                 dataset["international"], dataset["hall"]))

    # symmetrize cell
    # create symmetrized rotated cell from standard lattice and transformation matrix
    symmetrized_rotated_cell = np.dot(dataset['std_lattice'].T, dataset['transformation_matrix']).T
    # SVD decompose transformation from symmetrized cell to original cell to extract rotation
    # see https://igl.ethz.ch/projects/ARAP/svd_rot.pdf Eq. 17 and 21
    # S = X Y^T, X = symm_cell.T, Y = orig_cell.T
    (u, s, v_T) = np.linalg.svd(np.dot(symmetrized_rotated_cell.T, at.get_cell()))
    rotation = np.dot(v_T.T, u.T)
    # create symmetrized aligned cell by rotating symmetrized rotated cell
    symmetrized_aligned_cell = np.dot(rotation, symmetrized_rotated_cell.T).T
    # set new cell back to symmetrized aligned cell (approx. aligned with original directions, not axes)
    at.set_cell(symmetrized_aligned_cell, True)

    dataset = spglib.get_symmetry_dataset(at, symprec=symprec)
    orig_mapping = dataset['mapping_to_primitive']

    # create primitive cell
    (prim_cell, prim_scaled_pos, prim_types) = spglib.find_primitive(at, symprec=symprec)

    #rotate to align with orig cell
    prim_cell = np.dot(rotation, prim_cell.T).T
    prim_pos = np.dot(prim_scaled_pos, prim_cell)

    # align prim cell atom 0 with atom in orig cell that maps to it
    p = at.get_positions()
    dp0 = p[list(orig_mapping).index(0),:] - prim_pos[0,:]
    prim_pos += dp0

    # create symmetrized orig pos from prim cell pos + integer * prim cell lattice vectors
    prim_inv_cell = np.linalg.inv(prim_cell)
    for i in range(len(at)):
        dp_rounded = np.round( np.dot(p[i,:]-prim_pos[orig_mapping[i],:], prim_inv_cell) )
        p[i,:] = prim_pos[orig_mapping[i],:] + np.dot(dp_rounded, prim_cell)

    at.set_positions(p)

    # test final config with tight tol
    dataset = spglib.get_symmetry_dataset(at, symprec=1.0e-6)
    print("ase.calculators.symmetrize.refine_symmetry: precise ({}) "
          "symmetrized symmetry group number {}, "
          "international (Hermann-Mauguin) {} Hall {}"
          .format(1.0e-6, dataset["number"],
                  dataset["international"],
                  dataset["hall"]))
    return dataset

def check(at, symprec=1.0e-6):
    """
    Check symmetry of `at` with precision `symprec` using `spglib`

    Prints a summary and returns result of `spglib.get_symmetry_dataset()`
    """
    dataset = spglib.get_symmetry_dataset(at, symprec=symprec)
    print("ase.calculators.symmetrize.check: prec", symprec,
          "got symmetry group number", dataset["number"],
          ", international (Hermann-Mauguin)", dataset["international"],
          ", Hall ",dataset["hall"])
    return dataset

def prep(at, symprec=1.0e-6):
    """
    Prepare `at` for symmetry-preserving minimisation at precision `symprec`

    Returns a tuple `(rotations, translations, symm_map)`
    """
    dataset = spglib.get_symmetry_dataset(at, symprec=symprec)
    print("symmetry.prep: symmetry group number",dataset["number"],
          ", international (Hermann-Mauguin)", dataset["international"],
          ", Hall", dataset["hall"])
    rotations = dataset['rotations'].copy()
    translations = dataset['translations'].copy()
    symm_map=[]
    scaled_pos = at.get_scaled_positions()
    for (r, t) in zip(rotations, translations):
        this_op_map = [-1] * len(at)
        for i_at in range(len(at)):
            new_p = np.dot(r, scaled_pos[i_at,:]) + t
            dp = scaled_pos - new_p
            dp -= np.round(dp)
            i_at_map = np.argmin(np.linalg.norm(dp,  axis=1))
            this_op_map[i_at] = i_at_map
        symm_map.append(this_op_map)
    return (rotations, translations, symm_map)

def forces(lattice, inv_lattice, forces, rot, trans, symm_map):
    """
    Return symmetrized forces

    lattice vectors expected as row vectors (same as ASE get_cell() convention),
    inv_lattice is its matrix inverse (get_reciprocal_cell().T)
    """
    scaled_symmetrized_forces_T = np.zeros(forces.T.shape)

    scaled_forces_T = np.dot(inv_lattice.T,forces.T)
    for (r, t, this_op_map) in zip(rot, trans, symm_map):
        transformed_forces_T = np.dot(r, scaled_forces_T)
        scaled_symmetrized_forces_T[:,this_op_map[:]] += transformed_forces_T[:,:]
    scaled_symmetrized_forces_T /= len(rot)

    symmetrized_forces = np.dot(lattice.T, scaled_symmetrized_forces_T).T

    return symmetrized_forces

def stress(lattice, lattice_inv, stress_3_3, rot):
    """
    Return symmetrized stress

    lattice vectors expected as row vectors (same as ASE get_cell() convention),
    inv_lattice is its matrix inverse (get_reciprocal_cell().T)
    """
    scaled_stress = np.dot(np.dot(lattice, stress_3_3), lattice.T)

    symmetrized_scaled_stress = np.zeros((3,3))
    for r in rot:
        symmetrized_scaled_stress += np.dot(np.dot(r.T,scaled_stress),r)
    symmetrized_scaled_stress /= len(rot)

    return np.dot(np.dot(lattice_inv, symmetrized_scaled_stress),
                  lattice_inv.T)

class SymmetrizedCalculator(Calculator):
    """
    Calculator wrapper to preserve spacegroup symmetry during optimisation.

    Requires spglib package to be available.
    """
    implemented_properties = ['energy','forces','stress']
    def __init__(self, calc, atoms, *args, **kwargs):
        if not has_spglib:
            import spglib # generate ImportError to skip tests

        Calculator.__init__(self, *args, **kwargs)
        self.calc = calc
        (self.rotations, self.translations, self.symm_map) = prep(atoms)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            self.results = {}
        if 'energy' in properties and 'energy' not in self.results:
            self.results['energy'] = self.calc.get_potential_energy(atoms)
        if 'forces' in properties and 'forces' not in self.results:
            raw_forces = self.calc.get_forces(atoms)
            self.results['forces'] = forces(atoms.get_cell(),
                atoms.get_reciprocal_cell().T, raw_forces,
                self.rotations, self.translations, self.symm_map)
        if 'stress' in properties and 'stress' not in self.results:
            raw_stress = voigt_6_to_full_3x3_stress(self.calc.get_stress(atoms))
            symmetrized_stress = stress(atoms.get_cell(),
                                        atoms.get_reciprocal_cell().T,
                                        raw_stress, self.rotations)
            self.results['stress'] = full_3x3_to_voigt_6_stress(symmetrized_stress)
