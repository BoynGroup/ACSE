import sys
from pyscf import gto, scf, mcscf
import ACSE
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


#Set up system
mol = gto.M(atom=f"B 0 0 0; H 2 0 0",basis="sto-3g",symmetry=False,verbose=0, charge=-1,spin=1)
hf = scf.RHF(mol)
hf.kernel()

#Create initial 1 and 2 RDMs. For HF initial RDMs use a filled active space (2e in 1o or 1e in 1o)
nact, nels = 2,3
mc = mcscf.CASSCF(hf, nact, nels)
mc.kernel()

#Set up ACSE
acse = ACSE.acse(mc)
acse.max_iter = 3000
acse.ActiveRotations = False #False Zeros residual matrices active orbitals
acse.reconstruction = 'V' # V/NY
acse.eps = 0.001 #Euler step size
acse.e_conv = 1e-6 #Converge if dE is below this value
acse.set_math(math="BlockedD2Simplified") #How to solve the ACSE equations? FullD3, FullD2, BlockedD2Simplified
acse.singlet = False #Speeds up BlockedD2Simplified for singlet systems
acse.kernel()

