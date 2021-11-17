"""Vasp calculator based on the GenericFileIOCalculator"""
from typing import Mapping, Any
from subprocess import check_call
from pathlib import Path

from ase.calculators.genericfileio import (CalculatorTemplate,
                                           GenericFileIOCalculator)


class VaspProfile:
    def __init__(self, argv):
        self.argv = argv

    def version(self):
        raise NotImplementedError('Cannot check VASP version.')

    def run(self, directory, inputfile, outputfile):
        with open(outputfile, 'w') as fd:
            check_call(self.argv, stdout=fd, cwd=directory)


class VaspTemplate(CalculatorTemplate):
    _label = 'vasp'

    def __init__(self,
                 name='vasp',
                 implemented_properties=[
                     'energy', 'free_energy', 'forces', 'stress', 'magmom'
                 ]):
        super().__init__(name, implemented_properties)

    def write_input(self, directory, atoms, parameters, properties):
        raise NotImplementedError

    def execute(self, directory, profile):
        raise NotImplementedError

    def read_results(self, directory) -> Mapping[str, Any]:
        raise NotImplementedError


class Vasp(GenericFileIOCalculator):
    """Class for doing VASP calculations.
    """
    def __init__(self, *, profile=None, directory='.', **kwargs):
        """Construct VASP-calculator object.
        """

        if profile is None:
            profile = VaspProfile(['vasp'])

        super().__init__(template=VaspTemplate(),
                         profile=profile,
                         directory=directory,
                         parameters=kwargs)