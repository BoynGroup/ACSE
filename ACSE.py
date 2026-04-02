# A.W. Schlimgen & D.P. Gibney 4/2025

import numpy as np
import acseTools # TODO decide what goes in tools, if anything
#from acseTools import timing
#from openfermion.linalg import wedge
from functools import reduce
import scipy
import time

class acse():
    def __init__(self, mc): # TODO clean up, make like pyscf
        # How much/how can we inherit the attributes from mc?
        # How to handle Hartree-Fock, mf, input?
        self.mc = mc
        self.singlet = False
        self.mol = mc.mol
        self.Na = self.mol.nelec[0]
        self.Nb = self.mol.nelec[1]
        self.verbose = mc.verbose
        self.stdout = mc.stdout
        self.reconstruction = 'V' # Default V, TODO figure out M
        self.norm_tol = 1e-5 # TODO just picked these randomly
        self.e_tol = 1e-5 # TODO
        self.eps = 1e-2
        self.e_conv = 1e-5
        self.max_iter = 5000
        self.e_last = 0.
        self.last_norm = 1000000000000
        self.NO = False
        # Restart
        self.restart_files = None
        
        self.fcore = 0
        self.ncore = mc.ncore
        self.ncas = mc.ncas # when is this a tuple? TODO
        self.nelecas = mc.nelecas # when is this a tuple? TODO 
        self.N = mc.mol.nelectron - 2*self.fcore # when is this a tuple? frozen core? TODO
        self.r = mc.mol.nao_nr() - self.fcore  # Frozen core? TODO
        self.nvirt = mc.mol.nao_nr() - mc.ncas - mc.ncore # when is this a tuple TODO
        self.no = mc.mo_coeff # TODO should we change no to self.mo_coeff for consistency with pyscf?
        
        self.e_acse = 0.
        self.e_tot = self.e_acse + mc.mol.energy_nuc()
        self.hcore = mc.get_hcore() #mc.mol.intor_symmetric('int1e_kin') + mc.mol.intor_symmetric('int1e_nuc')

        self.rdms = None # Stored as ((d1a, d1b),(d2aa,d2ab,d2bb))

        self.datatype = np.float64
        self.ActiveRotations = False

    def make_rdm12s(self, mc): # TODO check compatability with pyscf, cf block, shci
        return self.rdms

    def get_energy_blockD2(self,D2):  # TODO factor of half in aa/bb part?
        E_aa = np.einsum('ijkl,ijkl',self.K2,D2[0])
        E_ab = np.einsum('ijkl,ijkl',self.K2,D2[1])*2
        E_bb = np.einsum('ijkl,ijkl',self.K2,D2[2])

        return E_aa + E_ab + E_bb

    def make_K2(self): # TODO shorten loop, einsum?
        ei1 = reduce(np.dot, (self.no.T, self.hcore, self.no))
        r = self.r
        ei2 = self.mc.mol.ao2mo(self.no, compact=False).reshape(r,r,r,r).swapaxes(1,2) # TODO Check alpha,beta for UHF?
        K2 = np.zeros(ei2.shape)
        N = self.N
        for i in range(r):
            for j in range(r):
                for k in range(r):
                    for l in range(r):
                        jl = 1. if j == l else 0.
                        ik = 1. if i == k else 0.
                        K2[i,j,k,l] = (ei1[j,l]*ik + ei1[i,k]*jl)/(N-1.) + ei2[i,j,k,l]
        return K2

    def make_K2_spinless_compact(self): # TODO shorten loop, einsum?
        #order is e1^dagger e1 e2^dagger e2
        #Not sure how to alter the order to fit our current formatting of e1^dagger e2^dagger e1 e2
        ei1 = reduce(np.dot, (self.no.T, self.hcore, self.no))
        r = self.r
        ei2 = self.mc.mol.ao2mo(self.no, compact=True) # TODO Check alpha,beta for UHF?
        N = self.N
        ijs = np.tril_indices(r)
        kls = np.tril_indices(r)
        for tmp1 in range(len(ijs[0])):
            for tmp2 in range(len(kls[0])):
                i = ijs[0][tmp1]
                k = ijs[1][tmp1]
                j = kls[0][tmp2]
                l = kls[1][tmp2]
                jl = 1. if j == l else 0.
                ik = 1. if i == k else 0.
                ei2[tmp1,tmp2] = (ei1[j,l]*ik + ei1[i,k]*jl)/(N-1.) + ei2[tmp1,tmp2]
        K2 = ei2.copy()
        return K2

    def make_fullK2_from_spinless_compact(self,K2_spinless_compact):
        r = self.r
        fullK2 = np.zeros((r*2,)*4)

        ijs = np.tril_indices(r)
        kls = np.tril_indices(r)
        for tmp1 in range(len(ijs[0])):
            for tmp2 in range(len(kls[0])):
                i = ijs[0][tmp1]
                j = ijs[1][tmp1]
                k = kls[0][tmp2]
                l = kls[1][tmp2]
                fullK2[i,j,k,l] = K2_spinless_compact[tmp1,tmp2]
                fullK2[j,i,l,k] = K2_spinless_compact[tmp1,tmp2]
                fullK2[l,k,i,j] = K2_spinless_compact[tmp1,tmp2]
                fullK2[k,l,j,i] = K2_spinless_compact[tmp1,tmp2]

                #bb
                fullK2[i+r,j+r,k+r,l+r] = K2_spinless_compact[tmp1,tmp2]
                fullK2[j+r,i+r,l+r,k+r] = K2_spinless_compact[tmp1,tmp2]
                fullK2[l+r,k+r,i+r,j+r] = K2_spinless_compact[tmp1,tmp2]
                fullK2[k+r,l+r,j+r,i+r] = K2_spinless_compact[tmp1,tmp2]

                #ab  TODO: verify this is valid for non-singlet systems
                fullK2[i,j,k+r,l+r] = K2_spinless_compact[tmp1,tmp2]/2 # ab
                fullK2[j,i,l+r,k+r] = K2_spinless_compact[tmp1,tmp2]/2 # ab
                fullK2[l,k,i+r,j+r] = K2_spinless_compact[tmp1,tmp2]/2 # ab
                fullK2[k,l,j+r,i+r] = K2_spinless_compact[tmp1,tmp2]/2 # ab

                fullK2[i,l+r,k+r,j] = -K2_spinless_compact[tmp1,tmp2]/2 # ab
                fullK2[j,k+r,l+r,i] = -K2_spinless_compact[tmp1,tmp2]/2 # ab
                fullK2[l,j+r,i+r,k] = -K2_spinless_compact[tmp1,tmp2]/2 # ab
                fullK2[k,i+r,j+r,l] = -K2_spinless_compact[tmp1,tmp2]/2 # ab
                
                fullK2[i+r,j+r,k,l] = K2_spinless_compact[tmp2,tmp1]/2 # ba
                fullK2[j+r,i+r,l,k] = K2_spinless_compact[tmp2,tmp1]/2 # ba
                fullK2[l+r,k+r,i,j] = K2_spinless_compact[tmp2,tmp1]/2 # ba
                fullK2[k+r,l+r,j,i] = K2_spinless_compact[tmp2,tmp1]/2 # ba

                fullK2[i+r,l,k,j+r] = -K2_spinless_compact[tmp2,tmp1]/2 # ab
                fullK2[j+r,k,l,i+r] = -K2_spinless_compact[tmp2,tmp1]/2 # ab
                fullK2[l+r,j,i,k+r] = -K2_spinless_compact[tmp2,tmp1]/2 # ab
                fullK2[k+r,i,j,l+r] = -K2_spinless_compact[tmp2,tmp1]/2 # ab
        
        return fullK2.transpose(0,2,1,3)

    def get_block_D1_from_block_D2(self,D2):
        Da = 2*np.einsum('ijkj->ik', D2[1])/(self.Nb)
        Db = 2*np.einsum('jijk->ik', D2[1])/(self.Na)

        Da_ = 2*np.einsum('ijkj->ik', D2[0])/(self.Na-1)
        Db_ = 2*np.einsum('jijk->ik', D2[2])/(self.Nb-1)

        Da__ = 2*np.einsum('jijk->ik', D2[0])/(self.Na-1)
        Db__ = 2*np.einsum('ijkj->ik', D2[2])/(self.Nb-1)
        
        Da = (Da + Da_ + Da__)/3
        Db = (Db + Db_ + Db__)/3

        #print(np.linalg.norm(Da-Da_))
        #print(np.linalg.norm(Db-Db_))
        return [Da,Db]

    def get_fullA_norms(self, A):
        r = self.r
        Aaa = A[0:r,0:r,0:r,0:r]
        Aab = (A[0:r,r::,0:r,r::])
        Abb = A[r::,r::,r::,r::]

        n = np.linalg.norm(A)
        naa = np.linalg.norm(Aaa.reshape(self.r**2, self.r**2))
        nab = np.linalg.norm(Aab.reshape(self.r**2, self.r**2))
        nbb = np.linalg.norm(Abb.reshape(self.r**2, self.r**2))

        return (n, naa, nab, nbb)

    def get_blockA_norms(self, A):
        r = self.r
        naa = np.linalg.norm(A[0])
        nab = np.linalg.norm(A[1])
        nbb = np.linalg.norm(A[2])
        n = naa + nab + nbb

        return (n, naa, nab, nbb)

    def active_space_EI(e1, e2, orbs, aorbs, mode='out'):
        numas = aorbs[1] - aorbs[0] + 1
        nas = aorbs[0] - 1
        e1new = np.zeros(shape=(numas,numas))
        e2new = e2[aorbs[0] - 1:aorbs[1], aorbs[0] - 1:aorbs[1], aorbs[0] - 1:aorbs[1], aorbs[0] - 1:aorbs[1]]
        # check if actually c or 0
        # numas = aorbs[1]-aorbs[0] + 1
        # numc = aorbs[0]-1
        # e1c0 = np.zeros(shape=(numc,numc))
        for p in range(0, numas):
            for s in range(0, numas):
                e1new[p,s] += e1[p + nas, s + nas]
                for i in range(0, nas):
                    e1new[p,s] += (2*e2[p + nas, i, s + nas, i] - e2[p + nas, i, i, s+nas])

    def set_math(self,math="BlockedD2Simplified"):
        '''Valid options
        FullD3
        FullD2
        BlockedD2Simplified'''

        if math == "FullD3":
            from D2 import make_fullD2, make_fullK2, get_energy_fullD2, get_acse_residual_full
            from D3 import reconstruct_fullD3
            def residual(D1,D2,K2):
                fullD1 = np.zeros((2*self.r,2*self.r),dtype=D1[0].dtype)
                fullD1[0:self.r,0:self.r] += D1[0]
                fullD1[self.r:,self.r:] += D1[1]

                fullD2 = make_fullD2(D2)
                fullK2 = make_fullK2(K2[0])
                fullD3 = reconstruct_fullD3(fullD1,fullD2,self.N,self.nelecas,self.mc.ncore,reconstruction=self.reconstruction)
                #from pyscf import fci
                #from pyscf.fci.addons import civec_spinless_repr
                #ci_spinless = civec_spinless_repr([self.mc.ci,], self.r, [self.mc.nelecas,])
                #d1, d2, d3 = fci.direct_spin1.make_rdm123(ci_spinless, self.r*2, (self.mc.nelecas[0] + self.mc.nelecas[1],0))
                #fullD3 = d3.transpose(0,2,4,1,3,5)/6
                residual = get_acse_residual_full(fullD2,fullD3,fullK2,residual=True)
                aa = residual[0:self.r,0:self.r,0:self.r,0:self.r]
                ab = residual[0:self.r,self.r:,0:self.r,self.r:]
                bb = residual[self.r:,self.r:,self.r:,self.r:]
                return [aa,ab,bb]
            def update(D1,D2,A):
                fullD1 = np.zeros((2*self.r,2*self.r),dtype=D1[0].dtype)
                fullD1[0:self.r,0:self.r] += D1[0]
                fullD1[self.r:,self.r:] += D1[1]

                fullD2 = make_fullD2(D2)
                fullA = make_fullD2(A)
                fullD3 = reconstruct_fullD3(fullD1,fullD2,self.N,self.nelecas,self.mc.ncore,reconstruction=self.reconstruction)
                A = get_acse_residual_full(fullD2,fullD3,fullA,residual=False)
                aa = A[0:self.r,0:self.r,0:self.r,0:self.r]
                ab = A[0:self.r,self.r:,0:self.r,self.r:]
                bb = A[self.r:,self.r:,self.r:,self.r:]
                return [aa,ab,bb]
            self.get_residual = residual
            self.get_update = update

        elif math == "FullD2":
            from D2 import make_fullD2, make_fullK2, get_acse_residual_full_no_D3
            def residual(D1,D2,K2):
                fullD1 = np.zeros((2*self.r,2*self.r),dtype=D1[0].dtype)
                fullD1[0:self.r,0:self.r] += D1[0]
                fullD1[self.r:,self.r:] += D1[1]

                fullD2 = make_fullD2(D2)
                fullK2 = make_fullK2(K2[0])
                residual = get_acse_residual_full_no_D3(fullD1,fullD2,fullK2,self.nelecas,self.mc.ncore,reconstruction=self.reconstruction,residual=True)
                aa = residual[0:self.r,0:self.r,0:self.r,0:self.r]
                ab = residual[0:self.r,self.r:,0:self.r,self.r:]
                bb = residual[self.r:,self.r:,self.r:,self.r:]
                return [aa,ab,bb]
            def update(D1,D2,A):
                fullD1 = np.zeros((2*self.r,2*self.r),dtype=D1[0].dtype)
                fullD1[0:self.r,0:self.r] += D1[0]
                fullD1[self.r:,self.r:] += D1[1]

                fullD2 = make_fullD2(D2)
                fullA = make_fullD2(A)
                A = get_acse_residual_full_no_D3(fullD1,fullD2,fullA,self.nelecas,self.mc.ncore,reconstruction=self.reconstruction,residual=False)
                aa = A[0:self.r,0:self.r,0:self.r,0:self.r]
                ab = A[0:self.r,self.r:,0:self.r,self.r:]
                bb = A[self.r:,self.r:,self.r:,self.r:]
                return [aa,ab,bb]
            self.get_residual = residual
            self.get_update = update
        elif math == "BlockedD2Simplified":
            if self.singlet == False:
                print("Not Singlet")
                from D2 import get_acse_residual_blocks_no_D3
                def residual(D1,D2,M):
                    return get_acse_residual_blocks_no_D3(D1,D2,M,self.nelecas,self.mc.ncore,reconstruction=self.reconstruction)
                self.get_residual = residual
                def update(D1,D2,A):
                    return get_acse_residual_blocks_no_D3(D1,D2,A,self.nelecas,self.mc.ncore,reconstruction=self.reconstruction,A=False)
                self.get_update = update
            else:
                print("Singlet")
                from D2 import get_acse_residual_blocks_no_D3_singlet
                def residual(D1,D2,M):
                    return get_acse_residual_blocks_no_D3_singlet(D1,D2,M,self.nelecas,self.mc.ncore,reconstruction=self.reconstruction)
                self.get_residual = residual
                def update(D1,D2,A):
                    return get_acse_residual_blocks_no_D3_singlet(D1,D2,A,self.nelecas,self.mc.ncore,reconstruction=self.reconstruction,A=False)
                self.get_update = update

    def kernel(self):
        from D2 import get_energy_blockD2
        r = self.r
        K2 = self.make_K2()
        K2 = (K2 + K2.transpose(1,0,3,2) + K2.transpose(2,3,0,1) + K2.transpose(3,2,1,0))/4
        K2 = K2.astype(self.datatype, copy=False)
        K2s = [K2,K2/2,K2]
        self.rdms = acseTools.make_full_rdms(self.mc) # Stored as ((d1a, d1b),(d2aa,d2ab,d2bb))

        D1 = self.rdms[0]
        D1[0] = D1[0].astype(self.datatype,copy=False)
        D1[1] = D1[1].astype(self.datatype,copy=False)
        D2 = self.rdms[1]
        D2[0] = D2[0].astype(self.datatype,copy=False)
        D2[1] = D2[1].astype(self.datatype,copy=False)
        D2[2] = D2[2].astype(self.datatype,copy=False)

        # Make A, get norm, and RDMs 
        norm = 100.
        it = 1
        e_diff = 100000
        
        if self.N >= 2:
            D2[0] /= 2.
            D2[1] /= 2.
            D2[2] /= 2.

        D2[0] = (D2[0] + D2[0].transpose(1,0,3,2) - D2[0].transpose(0,1,3,2) - D2[0].transpose(1,0,2,3))/4
        D2[0] = (D2[0] + D2[0].transpose(2,3,0,1))/2
        #D2[1] = (D2[1] + D2[1].transpose(1,0,3,2))/2
        D2[1] = (D2[1] + D2[1].transpose(2,3,0,1))/2
        D2[2] = (D2[2] + D2[2].transpose(1,0,3,2) - D2[2].transpose(0,1,3,2) - D2[2].transpose(1,0,2,3))/4
        D1 = self.get_block_D1_from_block_D2(D2)

        if self.restart_files is not None:
            fnames = self.restart_files
            D2[0] = np.load(fnames[0])
            D2[1] = np.load(fnames[1])
            D2[2] = np.load(fnames[2])
            D1 = self.get_block_D1_from_block_D2(D2)

        self.e_acse = get_energy_blockD2(D2, K2)    
        self.e_tot = self.e_acse + self.mc.mol.energy_nuc()
        self.last_e = 10000

        print("Active active rotations?: ",self.ActiveRotations)
        print("Reconstruction: ",self.reconstruction)
        print(self.e_tot, "Initial Energy") 
        print("Iteration : Norms : Energy")
        while abs(norm) >= self.norm_tol:
            start_time = time.perf_counter()

        
            A = self.get_residual(D1,D2,K2s)

            #Should this occur before or after zeroing?
            n = self.get_blockA_norms(A)
            if not self.ActiveRotations:
                active = slice(self.mc.ncore,self.mc.ncas+self.mc.ncore)
                A[0][active,active,active,active] = np.zeros((self.mc.ncas,)*4)
                A[1][active,active,active,active] = np.zeros((self.mc.ncas,)*4)
                A[2][active,active,active,active] = np.zeros((self.mc.ncas,)*4)

            #Updating D2
            D2_update = self.get_update(D1,D2,A)
            D2[0] += D2_update[0]*self.eps
            D2[1] += D2_update[1]*self.eps
            D2[2] += D2_update[2]*self.eps

            if (self.reconstruction in ['M']) or (self.NO):
               print('Rotating into Natural orbital Basis')
               NOs, NO_coeff = scipy.linalg.eigh(D1[0]+D1[1])#+np.diag([(x+1)*2 for x in range(D1[0].shape[0])]))
               sorted_indices = np.argsort(NOs)[::-1]
               #NOs -= np.array([(x+1)*2 for x in range(D1[0].shape[0])],dtype=float)
               NOs = NOs[sorted_indices]
               NO_coeff = NO_coeff[:, sorted_indices]
               #D1[0] = np.diag(NOs)
               D1[0] = NO_coeff.T @ D1[0] @ NO_coeff
               D1[1] = NO_coeff.T @ D1[1] @ NO_coeff
               #D1[0] = np.diag(np.diagonal(D1[0]))
               #D1[1] = np.diag(np.diagonal(D1[1]))
               #print("Unitary?")
               #print(NO_coeff @ NO_coeff.T)
               print(D1[0]+D1[1])
               #for i in range(NO_coeff.shape[0]):
               #    if NO_coeff[i,i] < 0:
               #       NO_coeff[i,:] *= -1
               #print(NO_coeff)
               #NO_coeff = np.linalg.inv(NO_coeff.T)
               #NO_coeff = NO_coeff.T
               #NO_coeff = np.identity(len(NOs))
                
               from pyscf import ao2mo
               def ao2mo_(g_ao, C):
                   g_1 = np.einsum("ds, abcd -> abcs", C, g_ao)
                   g_2 = np.einsum("cr, abcs -> abrs", C, g_1)
                   g_3 = np.einsum("bq, abrs -> aqrs", C, g_2)
                   g_mo = np.einsum("ap, aqrs -> pqrs", C, g_3)
                   return g_mo

               #K2_ = ao2mo.general(K2s[0],NO_coeff,aosym='s1')
               K2s[0] = ao2mo_(K2s[0],NO_coeff)
               K2s[1] = ao2mo_(K2s[1],NO_coeff)
               K2s[2] = ao2mo_(K2s[2],NO_coeff)
               K2 = ao2mo_(K2,NO_coeff)
               K2 = (K2 + K2.transpose(1,0,3,2) + K2.transpose(2,3,0,1) + K2.transpose(3,2,1,0))/4
               K2 = K2.astype(self.datatype, copy=False)
               K2s = [K2,K2/2,K2]
               D2[0] = ao2mo_(D2[0],NO_coeff)
               D2[1] = ao2mo_(D2[1],NO_coeff)
               D2[2] = ao2mo_(D2[2],NO_coeff)
            #Making sure D2 obeys symmetries/antisymmetries it should
            D2[0] = (D2[0] + D2[0].transpose(1,0,3,2) - D2[0].transpose(0,1,3,2) - D2[0].transpose(1,0,2,3))/4
            D2[0] = (D2[0] + D2[0].transpose(2,3,0,1))/2
            #D2[1] = (D2[1] + D2[1].transpose(1,0,3,2))/2
            D2[1] = (D2[1] + D2[1].transpose(2,3,0,1))/2
            D2[2] = (D2[2] + D2[2].transpose(1,0,3,2) - D2[2].transpose(0,1,3,2) - D2[2].transpose(1,0,2,3))/4
            D2[2] = (D2[2] + D2[2].transpose(2,3,0,1))/2
            D1 = self.get_block_D1_from_block_D2(D2)
            self.e_acse = get_energy_blockD2(D2,K2)
            self.e_tot = self.e_acse + self.mc.mol.energy_nuc()


            print("Iter %3.0f |A|(aa,ab,bb) = %.12f %.12f %.12f E = %.14f Time(s) %8.3f" % (it,n[1],n[2],n[3],self.e_tot,time.perf_counter() - start_time))
            print("Tr(Daa): %.7f Tr(Dab): %.7f Tr(Dbb): %.7f Tr(Da): %.7f Tr(Db): %.7f" % (np.trace(D2[0].reshape(r**2,r**2)),np.trace(D2[1].reshape(r**2,r**2)),np.trace(D2[2].reshape(r**2,r**2)),np.trace(D1[0]),np.trace(D1[1])))

            it += 1

            if self.e_tot > self.e_last:
                print("Energy Increased! Quitting")
                break
            if n[0] > self.last_norm:
                print('Norm Increased! Quitting')
                break
            if it >= self.max_iter:
                print("Max iterations reached! Quitting!")
                break
            if (self.last_e - self.e_tot) <= self.e_conv:
                print('Change in energy below convergence threshold! Quitting!')
                break

            if it % 50 == 0:
                np.save("d2aa.npy",D2[0])
                np.save("d2ab.npy",D2[1])
                np.save("d2bb.npy",D2[2])
        
            
            # TODO Update necessary things
            self.e_last = self.e_tot
            self.last_norm = n[0]
            self.last_e = self.e_tot
            self.D1 = D1
            self.D2 = D2
