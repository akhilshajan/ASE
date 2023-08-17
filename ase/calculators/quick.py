import numpy as np

from shutil import which
from ase.atoms import Atoms
from ase.calculators.calculator import FileIOCalculator
from ase.units import Bohr, Hartree

class QUICK(FileIOCalculator):
    implemented_properties = ['energy', 'forces', 'dipole']
    command = 'QUICK PREFIX.com' 
    discard_results_on_any_change = True

    default_parameters = {'charge': 0}

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='QUICK', atoms=None, scratch=None, ioplist=list(),
                 basisfile=None, extra=None, addsec=None, **kwargs):
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def calculate(self, *args, **kwargs):
        quick = ('quick', 'quick.cuda')
        if 'QUICK' in self.command:
            for qk in quick:
                if which(qk):
                    self.command = self.command.replace('QUICK', qk)
                    break
            else:
                raise EnvironmentError('Missing QUICK executable {}'
                                       .format(quick))

        FileIOCalculator.calculate(self, *args, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        atoms.write(self.label + '.com', format='xyz')
        charge = self.parameters.charge

        if 'method' in self.parameters:
            method = self.parameters['method'].upper()
        else:
            method = 'HF'

        if 'basis' in self.parameters:
            basis = self.parameters['basis'].upper()
        else:
            basis = 'STO-3G'

        if 'mult' in self.parameters:
            mult = self.parameters['mult'].upper()
        else:
            mult = '1'

        with open(self.label + '.com', 'r') as f:
            lines = f.readlines()
        lines[0] = str(method) + ' BASIS=' + str(basis) + ' TIGHTINT GRADIENT DIPOLE CHARGE=' + str(charge) + ' MULT=' + str(mult) + '\n' #str(atoms.get_initial_charges().sum())
        lines[1] = '\n'
        with open(self.label + '.com', 'w') as g:
            g.writelines(lines)
 
    def read_results(self):
        with open(self.label + '.out', 'r') as f:
            lines = f.readlines()
        geom_index = [x for x in range(len(lines)) if 'ANALYTICAL GRADIENT: ' in lines[x]][0] + 4
        charge_index = [x for x in range(len(lines)) if 'ATOMIC CHARGES' in lines[x]][0] + 2
        dipole_index = [x for x in range(len(lines)) if 'DIPOLE (DEBYE)' in lines[x]][0] + 2
        energy_index = [x for x in range(len(lines)) if 'TOTAL ENERGY' in lines[x]][0]
        elem = []
        mulliken = []
        lowdin = []
        # record elements and atomic charges
        while 'TOTAL' not in lines[charge_index]:
            e, m, l = lines[charge_index].split()
            elem.append(e)
            mulliken.append(float(m))
            lowdin.append(float(l))
            charge_index += 1
        # record coordinates and gradients
        coords = np.zeros([len(elem), 3])
        grads = np.zeros([len(elem), 3])
        i = 0
        readindex = geom_index + i
        while '----------------------------------------' not in lines[readindex]:
            lab, c, g = lines[readindex].split()
            atom_index = i // 3
            axis_index = i % 3
            coords[atom_index, axis_index] = float(c)
            grads[atom_index, axis_index] = float(g) * Hartree/Bohr
            i += 1
            readindex = geom_index + i
 
        # update the Atoms object
        self.atoms = Atoms(''.join(elem), positions=coords)
        self.atoms.charges = mulliken
        # record energy and forces
        self.results['energy'] = float(lines[energy_index].split()[-1])*Hartree
        self.results['forces'] = - grads
        self.results['dipole'] = np.array([float(x) for x in lines[dipole_index].split()[:3]]).reshape([1,3])
