import numpy as np
import scipy

def reconstruct_fullD3(fullD1,fullD2,N,nelecas,ncore,reconstruction='V'):
   '''Creates fullD3 matrix
   
      Args:
           fullD1: D1 matrix in the spin orbital basis (2r,2r)
           fullD2: D2 tensor in the spin orbital basis (2r,2r,2r,2r)
           N: Number of electrons
           nelecas: Number of electrons in the active space
           ncore: Number of core orbitals
           reconstruction: Which reconstruction to perform (V,NY)
      Returns:
           fullD3 in the spin representation (2r,2r,2r,2r,2r,2r) where r is the number of spatial orbitals
   '''
    r = fullD1.shape[0]//2
    nelecas = (nelecas[0]+ncore,nelecas[1]+ncore)

    #Obtain 2rdm cumulant
    d1wd1 = np.einsum('ik,jl->ijkl', fullD1, fullD1) - np.einsum('il,jk->ijkl', fullD1, fullD1)
    d1wd1 /= 2
    D2_c = (fullD2-d1wd1)*3
    M = D2_c + d1wd1

    #unsimplified 3-RDM reconstruction
    #fullD3_  = np.einsum('ijrq,pl->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('ijlq,pr->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('ijrl,pq->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('ijqr,pl->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('ijql,pr->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('ijrl,pq->ijprql',M,fullD1,optimize=True)
    #                                    
    #fullD3_ -= np.einsum('jirq,pl->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('jilq,pr->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('jirl,pq->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('jiqr,pl->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('jiql,pr->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('jirl,pq->ijprql',M,fullD1,optimize=True)
    #                                    
    #fullD3_ -= np.einsum('iprq,jl->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('iplq,jr->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('iprl,jq->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('ipqr,jl->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('ipql,jr->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('iprl,jq->ijprql',M,fullD1,optimize=True)
    #                                    
    #fullD3_ += np.einsum('jprq,il->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('jplq,ir->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('jprl,iq->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('jpqr,il->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('jpql,ir->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('jprl,iq->ijprql',M,fullD1,optimize=True)
    #                                    
    #fullD3_ -= np.einsum('pjrq,il->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('pjlq,ir->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('pjrl,iq->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('pjqr,il->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('pjql,ir->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('pjrl,iq->ijprql',M,fullD1,optimize=True)
    #                                    
    #fullD3_ += np.einsum('pirq,jl->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('pilq,jr->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('pirl,jq->ijprql',M,fullD1,optimize=True)
    #fullD3_ -= np.einsum('piqr,jl->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('piql,jr->ijprql',M,fullD1,optimize=True)
    #fullD3_ += np.einsum('pirl,jq->ijprql',M,fullD1,optimize=True)

    #fullD3_ /= 24

    ##Due to numerical errors building up every symmetry/antisymmetry
    ##needs to be explicity enforced. This is not all of them
    #fullD3_ = (fullD3_+fullD3_.transpose(1,2,0,3,4,5))/2
    #fullD3_ = (fullD3_+fullD3_.transpose(2,0,1,3,4,5))/2

    #fullD3_ = (fullD3_-fullD3_.transpose(1,0,2,3,4,5))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(0,2,1,3,4,5))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(2,1,0,3,4,5))/2

    #fullD3_ = (fullD3_-fullD3_.transpose(0,1,2,4,3,5))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(0,1,2,3,5,4))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(0,1,2,5,4,3))/2

    #fullD3_ = (fullD3_+fullD3_.transpose(0,1,2,4,5,3))/2
    #fullD3_ = (fullD3_+fullD3_.transpose(0,1,2,5,3,4))/2

    #fullD3_ = (fullD3_+fullD3_.transpose(3,4,5,0,1,2))/2
    #fullD3_ = (fullD3_+fullD3_.transpose(3,5,4,0,2,1))/2
    #fullD3_ = (fullD3_+fullD3_.transpose(5,3,4,2,0,1))/2
    #fullD3_ = (fullD3_+fullD3_.transpose(5,4,3,2,1,0))/2
    #fullD3_ = (fullD3_+fullD3_.transpose(4,5,3,1,2,0))/2
    #fullD3_ = (fullD3_+fullD3_.transpose(4,3,5,1,0,2))/2

    #fullD3_ = (fullD3_-fullD3_.transpose(3,4,5,1,0,2))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(3,5,4,2,0,1))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(5,3,4,0,2,1))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(5,4,3,1,2,0))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(4,5,3,2,1,0))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(4,3,5,0,1,2))/2

    #fullD3_ = (fullD3_-fullD3_.transpose(3,4,5,2,1,0))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(3,5,4,1,2,0))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(5,3,4,1,0,2))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(5,4,3,0,1,2))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(4,5,3,0,2,1))/2
    #fullD3_ = (fullD3_-fullD3_.transpose(4,3,5,2,0,1))/2
    #fullD3_ = (fullD3_+fullD3_.transpose(0,1,2,4,5,3))/2

    #1 term 8 transposes way of reconstructing the 3-RDM. Much greater satifaction of symmetry/antisymmetry conditions
    tmp_D3  = np.einsum('ijrq,pl->ijprql',M,fullD1,optimize=True)
    fullD3 = tmp_D3 + tmp_D3.transpose(2,0,1,3,4,5) + tmp_D3.transpose(1,2,0,3,4,5) - tmp_D3.transpose(0,1,2,3,5,4) - tmp_D3.transpose(2,0,1,3,5,4) - tmp_D3.transpose(1,2,0,3,5,4)
    fullD3 += tmp_D3.transpose(0,1,2,5,3,4) + tmp_D3.transpose(2,0,1,5,3,4) + tmp_D3.transpose(1,2,0,5,3,4)

    if N >= 3:
        fullD3 /= 9

    if reconstruction == 'NY': 
        D2_c /= 3
        #NY reconstruction from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.75.022505
        HF_occs = np.zeros(r*2)
        for i in range(r):
            if i < nelecas[0]:
                HF_occs[i] = 1
            else:
                HF_occs[i] = -1
            if i < nelecas[1]:
                HF_occs[i+r] = 1
            else:
                HF_occs[i+r] = -1

        tmp_  =1/6*np.einsum("l,ilrq,jpls->ijprqs",HF_occs,D2_c,D2_c,optimize=True)
        tmp  = tmp_ - tmp_.transpose(0,1,2,3,5,4) - tmp_.transpose(0,1,2,4,3,5) + tmp_.transpose(0,1,2,5,3,4) - tmp_.transpose(0,1,2,5,4,3) + tmp_.transpose(0,1,2,4,5,3)
        tmp += -tmp.transpose(0,2,1,3,4,5) -tmp.transpose(1,0,2,3,4,5) +tmp.transpose(2,0,1,3,4,5) -tmp.transpose(2,1,0,3,4,5) +tmp.transpose(1,2,0,3,4,5)

        fullD3 += tmp

    #if reconstruction == 'M': 
    #    D2_c /= 3 
    #    #M reconstruction from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.75.022505
    #    d1 = np.zeros((2*r))
    #    d1[:r] = scipy.linalg.eigh(fullD1[:r,:r])[0][::-1]
    #    d1[r:] = (scipy.linalg.eigh(fullD1[r:,r:])[0][::-1])
    #    NOs = np.einsum('i,j,k,q,s,t->ijkqst',d1,d1,d1,d1,d1,d1) - 3

    #    tmp_  =1/6*np.einsum("l,ilqs,jklt->ijkqst",HF_occs,D2_c,D2_c,optimize=True)
    #    M  = tmp_ - tmp_.transpose(0,1,2,3,5,4) - tmp_.transpose(0,1,2,4,3,5) + tmp_.transpose(0,1,2,5,3,4) - tmp_.transpose(0,1,2,5,4,3) + tmp_.transpose(0,1,2,4,5,3)
    #    M += -tmp.transpose(0,2,1,3,4,5) -M.transpose(1,0,2,3,4,5) +M.transpose(2,0,1,3,4,5) -M.transpose(2,1,0,3,4,5) +M.transpose(1,2,0,3,4,5)

    #    cumulant = np.divide(tmp,-NOs)
    #    fullD3 += cumulant/1

    return fullD3


