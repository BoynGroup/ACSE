import sys
sys.path.insert(0,'..')
from pyscf import gto, scf, mcscf, grad, ao2mo, fci
import ACSE, itertools
import numpy as np
import pytest
np.set_printoptions(threshold=sys.maxsize)

def test_reconstructions_with_active_propagation_FullD3_FullD2():
    max_iter = 10
    e_conv = 1e-6
    active_rotations = True
    #The M reconstruction only works for math=fullD3
    reconstruction = 'NY'
    #step size
    eps = 0.001

    distance = 1.3
    mol = gto.M(atom=f"B 0 0 0; H {distance} 0 0",basis="sto-3g",symmetry=False,verbose=0, charge=-1,spin=1)
    hf = scf.RHF(mol)
    hf.kernel()

    nact = 1
    nels = 1
        
    #mc = mcscf.CASCI(hf, mol.nao_nr(), mol.nelectron)#.fix_spin_(ss=1)
    #mc.fcisolver.nroots = 1
    #mc.kernel()

    mc = mcscf.CASSCF(hf, nact, nels)#.fix_spin_(ss=1)
    mc.kernel()

    print("FullD3")
    acse = ACSE.acse(mc)
    acse.max_iter = max_iter
    acse.ActiveRotations = active_rotations
    acse.reconstruction = reconstruction
    acse.eps = eps
    acse.e_conv = e_conv
    acse.set_math(math="FullD3")
    acse.kernel()
    FullD3_e = acse.e_tot

    acse = None

    print("FullD2")
    acse = ACSE.acse(mc)
    acse.max_iter = max_iter
    acse.ActiveRotations = active_rotations
    acse.reconstruction = reconstruction
    acse.eps = eps
    acse.e_conv = e_conv
    acse.set_math(math="FullD2")
    acse.kernel()
    FullD2_e = acse.e_tot
    assert FullD2_e == pytest.approx(FullD3_e,1e-9)

    #print("BlockedD2Simplified")
    #acse = ACSE.acse(mc)
    #acse.max_iter = max_iter
    #acse.datatype = np.float64
    #acse.ActiveRotations = active_rotations
    #acse.reconstruction = reconstruction
    #acse.eps = eps
    #acse.e_conv = e_conv
    #acse.set_math(math="BlockedD2Simplified")
    #acse.kernel()


def test_reconstructions_with_active_propagation_FullD2_BlockedD2Simplified():
    max_iter = 10
    e_conv = 1e-6
    active_rotations = True
    #The M reconstruction only works for math=fullD3
    reconstruction = 'NY'
    #step size
    eps = 0.001

    distance = 1.3
    mol = gto.M(atom=f"B 0 0 0; H {distance} 0 0",basis="sto-3g",symmetry=False,verbose=0, charge=-1,spin=1)
    hf = scf.RHF(mol)
    hf.kernel()

    nact = 1
    nels = 1
        
    mc = mcscf.CASSCF(hf, nact, nels)#.fix_spin_(ss=1)
    mc.kernel()

    print("FullD2")
    acse = ACSE.acse(mc)
    acse.max_iter = max_iter
    acse.datatype = np.float64
    acse.ActiveRotations = active_rotations
    acse.reconstruction = reconstruction
    acse.eps = eps
    acse.e_conv = e_conv
    acse.set_math(math="FullD2")
    acse.kernel()
    FullD3_e = acse.e_tot

    acse = None

    print("BlockedD2Simplified")
    acse = ACSE.acse(mc)
    acse.max_iter = max_iter
    acse.datatype = np.float64
    acse.ActiveRotations = active_rotations
    acse.reconstruction = reconstruction
    acse.eps = eps
    acse.e_conv = e_conv
    acse.set_math(math="BlockedD2Simplified")
    acse.kernel()
    FullD2_e = acse.e_tot
    assert FullD2_e == pytest.approx(FullD3_e,1e-9)


def test_reconstructions_with_active_propagation_BlockedD2Simplified_BlockedD2Simplified_Singlet():
    max_iter = 10
    e_conv = 1e-6
    active_rotations = True
    #The M reconstruction only works for math=fullD3
    reconstruction = 'NY'
    #step size
    eps = 0.001

    distance = 1.3
    mol = gto.M(atom=f"B 0 0 0; H {distance} 0 0",basis="sto-3g",symmetry=False,verbose=0, charge=0,spin=0)
    hf = scf.RHF(mol)
    hf.kernel()

    nact = 1
    nels = 2
        
    mc = mcscf.CASSCF(hf, nact, nels)#.fix_spin_(ss=1)
    mc.kernel()

    print("BlockedD2Simplified")
    acse = ACSE.acse(mc)
    acse.max_iter = max_iter
    acse.datatype = np.float64
    acse.ActiveRotations = active_rotations
    acse.reconstruction = reconstruction
    acse.eps = eps
    acse.e_conv = e_conv
    acse.set_math(math="FullD2")
    acse.kernel()
    FullD3_e = acse.e_tot

    acse = None

    print("BlockedD2Simplified Singlet")
    acse = ACSE.acse(mc)
    acse.max_iter = max_iter
    acse.datatype = np.float64
    acse.ActiveRotations = active_rotations
    acse.reconstruction = reconstruction
    acse.eps = eps
    acse.e_conv = e_conv
    acse.singlet = True
    acse.set_math(math="BlockedD2Simplified")
    acse.kernel()
    FullD2_e = acse.e_tot
    assert FullD2_e == pytest.approx(FullD3_e,1e-9)


