# A Python/PySCF implimentation of the Anti-Hermitian Contracted Schrodinger equation (ACSE)
Solves the ACSE in a spin blocked representation using either the Valdemoro (V) or Nakatsuji-Yasuda (NY) reconstructions of the 3-RDM. 


**Norms**
```math
\begin{align}
Tr({}^2D_{aa}) &= N_a(N_a-1)/2!\\
Tr({}^2D_{ab}) &= N_a(N_b)/2!  \\
Tr({}^2D_{bb}) &= N_b(N_b-1)/2!\\
Tr({}^1D_{a})  &= N_a/1!        \\
Tr({}^1D_{b})  &= N_b/1!         \\
\end{align}
```
