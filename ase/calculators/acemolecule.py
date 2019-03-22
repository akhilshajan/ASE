from __future__ import print_function
import os
from copy import deepcopy
#import sys
#from ase.atoms import Atoms
from ase.io.acemolecule import read_acemolecule_out
from ase.calculators.calculator import ReadError
from ase.calculators.calculator import FileIOCalculator
#import numpy as np


class ACE(FileIOCalculator):
    '''
    ACE-Molecule logfile reader
    It has default parameters of each input section
    And parameters' type = list of dictionaries
    '''


    name = 'ace'
    implemented_properties = ['energy', 'forces',
                              'geometry', 'excitation-energy']
    # defaults is default value of ACE-input
    basic_list = [{
        'Type': 'Scaling', 'Scaling': '0.35', 'Basis': 'Sinc',
                  'Grid': 'Sphere',
                  'KineticMatrix': 'Finite_Difference', 'DerivativesOrder': '7',
                  'GeometryFilename': None, 'NumElectrons': None}
                  ]
    scf_list = [{
        'ExchangeCorrelation': {'XFunctional': 'GGA_X_PBE', 'CFunctional': 'GGA_C_PBE'},
        'NumberOfEigenvalues': None,
    }]

    force_list = [{'ForceDerivative': 'Potential'}]
    tddft_list = [{
        'SortOrbital': 'Order', 'MaximumOrder': '10',
        'ExchangeCorrelation': {'XFunctional': 'GGA_X_PBE', 'CFunctional': 'GGA_C_PBE'},
    }]

    order_list = ['BasicInformation', 'Guess', 'Scf']
    guess_list = [{}]  # now not need this
    cis_list = [{}]
    cisd_list = [{}]
    dda_list = [{}]

    default_parameters = {'BasicInformation': basic_list, 'Guess': guess_list,
                          'Scf': scf_list, 'Force': force_list, 'TDDFT': tddft_list, 'order': order_list}
    parameters = default_parameters
    command = 'mpirun -np 1 ../ace PREFIX.inp > PREFIX.log'

    def __init__(
            self, restart=None, ignore_bad_restart_file=False,
            label='ace', atoms=None, command=None,
            basisfile=None, **kwargs):
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, command=command, **kwargs)

    def set(self, **kwargs):
        '''Change or add parameters and return changed parameter
            Parameters
            ==========
            kwargs: It is parameter that we want to change and type is dictionary
            
            
            Returns
            =======
            kwargs
        '''
        new_parameters = deepcopy(self.parameters)
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if 'order' in kwargs:
            new_parameters['order'] = kwargs['order']
            append_default_parameter = list(set(kwargs['order']))
            for value in append_default_parameter:
                # This is for adding default values of repeated section
                repeat = kwargs['order'].count(value)
                if repeat > 1:
                    for i in range(repeat - 1):
                        if value in self.default_parameters.keys():
                            new_parameters[value] += self.default_parameters[value]
        for key in new_parameters['order']:
            if key in kwargs.keys():      # key : BasicInformation, Force, Scf and so on
                if isinstance(kwargs[key], dict):
                    dict_to_list = []
                    dict_to_list.append(kwargs[key])
                    # kwargs[key] : basic_list, force_lsit ....
                    kwargs[key] = dict_to_list
                i = 0
                for val in kwargs[key]:
                    if len(new_parameters[key]) > 0:
                        new_parameters[key][i] = update_parameter(
                            new_parameters[key][i], val)

                    else:
                        new_parameters[key] = [val]
                    i = i + 1
        self.parameters = new_parameters
        return changed_parameters

    def read(self, label):
        FileIOCalculator.read(self, label)
        filename = self.label + ".log"
        if not os.path.isfile(filename):
            raise ReadError
        self.read_results()

    def make_xyz_file(self, atoms):
        '''Writes the xyz file '''
        atoms.write("{}_opt.xyz".format(self.label))

    def write_input(self, atoms, properties=None, system_changes=None):
        '''Writes the input file and xyz file. And if properties is ['forces'], add'Force' to parameters['order']
            
            Parameters
            ==========
            atoms : ASE atoms
            properties : one of Implement_properties, and type is list 
        
        '''
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        inputfile = open(self.label + '.inp', 'w')
        copy_parameters = deepcopy(self.parameters)
        if properties == ['forces'] and not 'Force' in copy_parameters['order']:
            copy_parameters['order'].append('Force')
        self.make_xyz_file(atoms)
        copy_parameters["BasicInformation"][0]["GeometryFilename"] = "{}_opt.xyz".format(
            self.label)
        copy_parameters["BasicInformation"][0]["GeometryFormat"] = "xyz"
        self.write_acemolecule_input(inputfile, copy_parameters)

        inputfile.close()

    def read_results(self):
        '''Read results from logfile. results consist of quantites. And default quantities = [energy, forces, atoms, excitation-energy]
            temporarily excitation-energy value is 1. This is becuase preferentially we want to calculate TDDFT by using ase.
        
        '''
        filename = self.label + '.log'
        f = open(filename, "r")
        tddft = len(f.read().split("TDDFT"))
        if tddft > 2:
            quantities = ['excitation-energy']
        else:
            quantities = ['energy', 'forces', 'atoms', 'excitation-energy']
        for value in quantities:
            self.results[value] = read_acemolecule_out(
                filename, quantity=value)

    def write_acemolecule_section(self, fpt, section, indent=0):
        '''Write parameters in each section of input 
            
            Parameters
            ==========
            fpt : ACE-Moleucle input file name
            section : self.parameters

        '''
        for key, val in section.items():
            if isinstance(val, str) or isinstance(val, int) or isinstance(val, float):
                fpt.write('    ' * indent + str(key) + " " + str(val) + "\n")
            elif isinstance(val, dict):
                fpt.write('    ' * indent + "%% " + str(key) + "\n")
                self.write_acemolecule_section(fpt, val, indent + 1)
                fpt.write('    ' * indent + "%% End\n")

        return

    def write_acemolecule_input(self, fpt, param, indent=0):
        '''Write ACE-Molecule input 

            Our input format Examples
            ========================= 
        %% BasicInformation
            Type	Scaling
            Scaling	0.4
            Basis	Sinc
            Cell	10.0
            Grid	Sphere
            GeometryFormat		xyz
            SpinMultiplicity	3.0
            Polarize    1
            Centered		0
            OccupationMethod		ZeroTemp
            %% Pseudopotential
                Pseudopotential	1
                UsingDoubleGrid 0
                FilterType Sinc
                Format	upf
                PSFilePath	/home/khs/DATA/UPF
                PSFileSuffix	.pbe-theos.UPF
            %% End
            GeometryFilename	xyz/C.xyz
        %% End
        %% Guess
        %% End
        %% Scf
            IterateMaxCycle	150
            ConvergenceType	Energy
            ConvergenceTolerance	0.00001
            EnergyDecomposition		1
            ComputeInitialEnergy		1
            %% Diagonalize
                DiagonalizeMaxIter	10
                Tolerance	0.000001
                FullOrthogonalize		1
            %% End
            %% ExchangeCorrelation
                XFunctional		GGA_X_PBE
                CFunctional		GGA_C_PBE
            %% End
            %% Mixing
                MixingMethod	1
                MixingType	Density
                MixingParameter	0.5
                PulayMixingParameter	0.1
            %% End
        %% End

         '''
        prefix = "    " * indent

        for i in range(len(param['order'])):
            fpt.write(prefix + "%% " + param['order'][i] + "\n")
            section_list = param[param['order'][i]]
            if len(section_list) > 0:
                section = section_list.pop(0)
                self.write_acemolecule_section(fpt, section, 1)
            fpt.write("%% End\n")
        return


def update_parameter(oldpar, newpar):
    '''Replace or add dict into oldpar from newpar 
        
        Parameters
        ==========
        oldpar : original parameters
        newpar : new dictionary that we want to change or add to origianl parameters.


        Return
        ======
        oldpar

    '''
    for key, val in newpar.items():
        if key in oldpar:
            if isinstance(val, dict):
                update_parameter(oldpar[key], val)
            else:
                oldpar.update(newpar)
        else:
            oldpar.update(newpar)

    return oldpar


if __name__ == "__main__":
    print('khs')
