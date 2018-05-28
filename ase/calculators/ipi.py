from __future__ import print_function
import socket
from subprocess import Popen

import numpy as np

from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)
import ase.units as units


class IPIProtocol:
    """Communication using IPI protocol."""

    statements = {'POSDATA', 'GETFORCE', 'STATUS', 'INIT', ''}
    # The statement '' means end program.
    responses = {'READY', 'HAVEDATA', 'FORCEREADY', 'NEEDINIT'}

    def __init__(self, socket, txt=None):
        self.socket = socket

        if txt is None:
            log = lambda *args: None
        else:
            def log(*args):
                print('IPI:', *args, file=txt)
                txt.flush()
        self.log = log

    def sendmsg(self, msg):
        self.log('  sendmsg', repr(msg))
        #assert msg in self.statements, msg
        msg = msg.encode('ascii').ljust(12)
        self.socket.sendall(msg)

    def recvmsg(self):
        msg = self.socket.recv(12)
        assert len(msg) == 12, msg
        msg = msg.rstrip().decode('ascii')
        #assert msg in self.responses, msg
        self.log('  recvmsg', repr(msg))
        return msg

    def send(self, a, dtype):
        buf = np.asarray(a, dtype).tobytes()
        #self.log('  send {}'.format(np.array(a).ravel().tolist()))
        self.log('  send {} bytes of {}'.format(len(buf), dtype))
        self.socket.sendall(buf)

    def recv(self, shape, dtype):
        a = np.empty(shape, dtype)
        nbytes = np.dtype(dtype).itemsize * np.prod(shape)
        buf = self.socket.recv(nbytes)
        assert len(buf) == nbytes, (len(buf), nbytes)
        self.log('  recv {} bytes of {}'.format(len(buf), dtype))
        #print(np.frombuffer(buf, dtype=dtype))
        a.flat[:] = np.frombuffer(buf, dtype=dtype)
        #self.log('  recv {}'.format(a.ravel().tolist()))
        assert np.isfinite(a).all()
        return a

    def sendposdata(self, cell, icell, positions):
        assert cell.size == 9
        assert icell.size == 9
        assert positions.size % 3 == 0

        self.log(' sendposdata')
        self.sendmsg('POSDATA')
        self.send(cell / units.Bohr, np.float64)
        self.send(icell * units.Bohr, np.float64)
        self.send(len(positions), np.int32)
        self.send(positions / units.Bohr, np.float64)

    def recvposdata(self):
        cell = self.recv((3, 3), np.float64)
        icell = self.recv((3, 3), np.float64)
        natoms = self.recv(1, np.int32)
        natoms = int(natoms)
        positions = self.recv((natoms, 3), np.float64)
        return cell * units.Bohr, icell / units.Bohr, positions * units.Bohr

    def sendrecv_force(self):
        self.log(' sendrecv_force')
        self.sendmsg('GETFORCE')
        msg = self.recvmsg()
        assert msg == 'FORCEREADY', msg
        e = self.recv(1, np.float64)[0]
        natoms = self.recv(1, np.int32)
        assert natoms >= 0
        forces = self.recv((int(natoms), 3), np.float64)
        virial = self.recv((3, 3), np.float64)
        nmorebytes = self.recv(1, np.int32)
        nmorebytes = int(nmorebytes)
        if nmorebytes > 0:
            # Receiving 0 bytes will block forever on python2.
            morebytes = self.recv(nmorebytes, np.byte)
        else:
            morebytes = b''
        return (e * units.Ha, (units.Ha / units.Bohr) * forces,
                units.Ha * virial, morebytes)

    def sendforce(self, energy, forces, virial,
                  morebytes=np.empty(0, dtype=np.byte)):
        assert np.array([energy]).size == 1
        assert forces.shape[1] == 3
        assert virial.shape == (3, 3)

        self.log(' sendforce')
        self.sendmsg('FORCEREADY')  # mind the units
        self.send(np.array([energy / units.Ha]), np.float64)
        natoms = len(forces)
        self.send(np.array([natoms]), np.int32)
        self.send(units.Bohr / units.Ha * forces, np.float64)
        self.send(1.0 / units.Ha * virial, np.float64)
        self.send(np.array([len(morebytes)]), np.int32)
        self.send(morebytes, np.byte)

    def status(self):
        self.log(' status')
        self.sendmsg('STATUS')
        msg = self.recvmsg()
        return msg

    def end(self):
        self.log( 'end')
        self.sendmsg('')

    def sendinit(self):
        # XXX Not sure what this function is supposed to send.
        # It 'works' with QE, but for now we try not to call it.
        self.log(' sendinit')
        self.sendmsg('INIT')
        self.send(0, np.int32)  # 'bead index' always zero
        # number of bits (don't they mean bytes?) in initialization string:
        # Why does quantum espresso seem to want -1?  Is that normal?
        self.send(-1, np.int32)
        self.send(np.empty(0), np.byte)  # initialization string

    def calculate(self, positions, cell):
        self.log('calculate')
        msg = self.status()
        # We don't know how NEEDINIT is supposed to work, but some codes
        # seem to be okay if we skip it and send the positions instead.
        #if msg == 'NEEDINIT':
        #    self.sendinit()
        #    msg = self.status()
        #assert msg == 'READY', msg
        icell = np.linalg.pinv(cell).transpose()
        self.sendposdata(cell, icell, positions)
        msg = self.status()
        assert msg == 'HAVEDATA', msg
        e, forces, virial, morebytes = self.sendrecv_force()
        r = dict(energy=e,
                 forces=forces,
                 virial=virial)
        if morebytes:
            r['morebytes'] = morebytes
        return r


class IPIServer:
    def __init__(self, process_args=None, host='localhost', port=31415,
                 socketfname=None, log=None):
        self.host = host
        self.port = port
        self.socketfname = socketfname
        self._closed = False

        if socketfname is not None:
            self.serversocket = socket.socket(socket.AF_UNIX)
            self.serversocket.bind(socketfname)
        else:
            self.serversocket = socket.socket(socket.AF_INET)
            self.serversocket.setsockopt(socket.SOL_SOCKET,
                                         socket.SO_REUSEADDR, 1)
            self.serversocket.bind((host, port))
        if log:
            print('Accepting IPI clients on {}:{}'.format(host, port),
                  file=log)
        self.serversocket.listen(1)

        self.log = log

        self.proc = None

        self.ipi = None
        self.clientsocket = None
        self.address = None

        if process_args is not None:
            if log:
                print('Launch subprocess: {}'.format(process_args), file=log)
            self.proc = Popen(process_args, shell=True)
            # self._accept(process_args)

    def _accept(self, process_args=None):
        # It should perhaps be possible for process to be launched by user
        log = self.log
        if self.log:
            print('Awaiting client', file=self.log)
        self.clientsocket, self.address = self.serversocket.accept()
        if log:
            print('Accepted connection from {}'.format(self.address), file=log)

        self.ipi = IPIProtocol(self.clientsocket, txt=log)

    def close(self):
        if self._closed:
            return

        self.ipi.log('Close IPI server')
        self._closed = True

        # Proper way to close sockets?
        if self.ipi is not None:
            self.ipi.end()  # Send end-of-communication string
            self.ipi = None
        if self.proc is not None:
            exitcode = self.proc.wait()
            if exitcode != 0:
                import warnings
                # Quantum Espresso seems to always exit with status 128,
                # even if successful.
                # Should investigate at some point
                warnings.warn('Subprocess exited with status {}'
                              .format(exitcode))
        if self.clientsocket is not None:
            self.clientsocket.shutdown(socket.SHUT_RDWR)
        if self.serversocket is not None:
            self.serversocket.shutdown(socket.SHUT_RDWR)
        #self.log('IPI server closed')

    def calculate(self, atoms):
        assert not self._closed

        # If we have not established connection yet, we must block
        # until the client catches up:
        if self.ipi is None:
            self._accept()
        return self.ipi.calculate(atoms.positions, atoms.cell)


class IPIClient:
    def __init__(self, host='localhost', port=31415,
                 socketfname=None, log=None):
        self.host = host
        self.port = port
        self.socketfname = socketfname

        if socketfname is not None:
            sock = socket.socket(socket.AF_UNIX)
            sock.connect(socketfname)
        else:
            sock = socket.socket(socket.AF_INET)
            sock.connect((host, port))
        self.ipi = IPIProtocol(sock, txt=log)
        self.log = self.ipi.log
        self.closed = False

        self.state = 'READY'

    def close(self):
        if not self.closed:
            self.closed = True
            self.ipi.socket.close()

    def irun(self, atoms, use_stress=True):
        try:
            while True:
                msg = self.ipi.recvmsg()
                if msg == '':
                    self.close()
                    return
                elif msg == 'STATUS':
                    self.ipi.sendmsg(self.state)
                elif msg == 'POSDATA':
                    assert self.state == 'READY'
                    cell, icell, positions = self.ipi.recvposdata()
                    atoms.cell[:] = cell
                    atoms.positions[:] = positions
                    # User may wish to do something with the atoms object now.
                    yield
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                    if use_stress:
                        stress = atoms.get_stress(voigt=False)
                        virial = -atoms.get_volume() * stress
                    else:
                        virial = np.zeros((3, 3))
                    self.state = 'HAVEDATA'
                elif msg == 'GETFORCE':
                    assert self.state == 'HAVEDATA', self.state
                    self.ipi.sendforce(energy, forces, virial)
                    self.state = 'READY'
        finally:
            self.close()

    def run(self, atoms, use_stress=True):
        for _ in self.irun(atoms, use_stress=use_stress):
            pass


class IPICalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    ipi_supported_changes = {'positions', 'cell'}

    def __init__(self, calc=None, host='localhost', port=31415,
                 socketfname=None, log=None):
        Calculator.__init__(self)
        self.calc = calc
        self.host = host
        self.port = port
        self.socketfname = socketfname
        self.ipi = None
        self.log = log

    def todict(self):
        d = {'type': 'calculator',
                'name': 'ipi'}
        if self.calc is not None:
            d['calc'] = self.calc.todict()
        return d

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        bad = [change for change in system_changes
               if change not in self.ipi_supported_changes]

        if self.ipi is not None and any(bad):
            raise PropertyNotImplementedError(
                'Cannot change {} through IPI protocol.  '
                'Please create new IPI calculator.'
                .format(bad if len(bad) > 1 else bad[0]))

        if self.ipi is None:
            if self.calc is not None:
                cmd = self.calc.command
                cmd = cmd.format(host=self.host, port=self.port,
                                 socketfname=self.socketfname,
                                 prefix=self.calc.prefix)
                self.calc.write_input(atoms, properties=properties,
                                      system_changes=system_changes)
            else:
                cmd = None  # User configures/launches subprocess
                # (and is responsible for having generated any necessary files)
            self.ipi = IPIServer(process_args=cmd, port=self.port,
                                 socketfname=self.socketfname,
                                 host=self.host, log=self.log)

        self.atoms = atoms.copy()
        results = self.ipi.calculate(atoms)
        virial = results.pop('virial')
        vol = atoms.get_volume()
        from ase.constraints import full_3x3_to_voigt_6_stress
        results['stress'] = -full_3x3_to_voigt_6_stress(virial) / vol
        self.results.update(results)

    def close(self):
        if self.ipi is not None:
            self.ipi.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
