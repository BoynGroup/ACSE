# A Python/PySCF implimentation of the Anti-Hermitian Contracted Schrodinger equation (ACSE)
Solves the ACSE in a spin blocked representation using either the Valdemoro (V) or Nakatsuji-Yasuda (NY) reconstructions of the 3-RDM.

### 3rdm cumulant decomposition
```math
{}^3D = (3^2\Delta + ({}^1D \wedge {}^1D)) \wedge {}^1D + {}^3\Delta
```

### Valdemoro (V)
```math
{}^3\Delta = 0
```

### Nakatsuji-Yasuda (NY)
```math
{}^3\Delta^{ijp}_{rql} = \frac{1}{6}\sum_{a}\sigma_a\hat{A}({}^2\Delta^{ia}_{rq}{}^2\Delta^{jp}_{al})
```

### Installation
1. **Clone Repo**
```bash
git clone https://github.com/BoynGroup/ACSE.git
cd ACSE
```

2. **Create Python Virtual Environment and Activate it**
```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install ACSE**
```bash
install ACSE and dependencies
python -m pip install -e ACSE/Location
```

### Usage
```bash
import sys
from pyscf import gto, scf, mcscf
import ACSE
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


#Set up system
mol = gto.M(atom=f"B 0 0 0; H 2 0 0",basis="sto-3g",symmetry=False,verbose=0, charge=-1,spin=1)
hf = scf.RHF(mol)
hf.kernel()

#Create initial 1 and 2 RDMs. For HF initial RDMs use a filled active space (2e in 1o)
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
```

**RDM Normalizations**
```math
\begin{align}
Tr({}^2D_{aa}) &= N_a(N_a-1)/2!\\
Tr({}^2D_{ab}) &= N_a(N_b)/2!  \\
Tr({}^2D_{bb}) &= N_b(N_b-1)/2!\\
Tr({}^1D_{a})  &= N_a/1!        \\
Tr({}^1D_{b})  &= N_b/1!         \\
\end{align}
```

### Publications
[Initial publication](https://arxiv.org/pdf/2604.02550)
