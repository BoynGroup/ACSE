import numpy as np

def get_energy_fullD2(fullD2, fullK2):  # TODO factor of half in aa/bb part?
    return np.einsum('ijkl,ijkl',fullK2,fullD2)

def make_fullD2(rdm2): #Assumes rdm2 is of the shape (d2aa,d2ab,d2bb)
     '''Creates fullD2 matrix
     
        Args:
             rdm2: Spin orbital 2RDM of shape (aa,ab,bb)
        Returns:
             D2 in the spin representation (2r,2r,2r,2r) where r is the number of spatial orbitals
     '''
     d2aa, d2ab, d2bb = rdm2
     r = d2aa.shape[0]
     fullD2 = np.zeros((2*r,2*r,2*r,2*r))
     for i in range(r):
         for j in range(r):
             for k in range(r):
                 for l in range(r):
     
                     fullD2[i,j,k,l] = d2aa[i,j,k,l] # aa
                     fullD2[i+r,j+r,k+r,l+r] = d2bb[i,j,k,l] # bb
                     fullD2[i,j+r,k,l+r] = d2ab[i,j,k,l]/1 # ab
                     fullD2[i,j+r,l+r,k] = -d2ab[i,j,k,l]/1
                     fullD2[i+r,j,k+r,l] = d2ab[j,i,l,k]/1 # ba
                     fullD2[i+r,j,l,k+r] = -d2ab[j,i,l,k]/1 # ba
     return fullD2

def make_fullK2(K2):
   '''Creates fullK2 matrix
   
      Args:
           K2: Spatial orbital K2
      Returns:
           K2 in the spin representation (2r,2r,2r,2r) where r is the number of spatial orbitals
   '''
   # TODO factor of half in ab/ba blocks? 
   r = K2.shape[0]
   fullK2 = np.zeros((2*r,2*r,2*r,2*r))
   for i in range(r):
       for j in range(r):
           for k in range(r):
               for l in range(r):
   
                   fullK2[i,j,k,l] = K2[i,j,k,l] # aa
                   fullK2[i+r,j+r,k+r,l+r] = K2[i,j,k,l] # bb
                   fullK2[i,j+r,k,l+r] = K2[i,j,k,l]/2 # ab
                   fullK2[i,j+r,l+r,k] = -K2[i,j,k,l]/2
                   fullK2[i+r,j,k+r,l] = K2[j,i,l,k]/2 # ba
                   fullK2[i+r,j,l,k+r] = -K2[j,i,l,k]/2 # ba
   return fullK2


def get_acse_residual_full(fullD2,fullD3,M,residual=True):
    # J. Chem. Phys. 126, 184101 2007
    #Non-simplified D2 terms 
    #A2  =-np.einsum('ijqp,klqp->ijkl',fullD2,M,optimize="optimal")*1.
    #A2 += np.einsum('ijqp,lkqp->ijkl',fullD2,M,optimize="optimal")*1.
    #A2 += np.einsum('pqlk,pqji->ijkl',fullD2,M,optimize="optimal")*1.
    #A2 -= np.einsum('pqlk,pqij->ijkl',fullD2,M,optimize="optimal")*1.

    #First simplification
    #A2  =-np.einsum('ijqp,klqp->ijkl',fullD2,M,optimize="optimal")*2.
    #A2 += np.einsum('pqlk,pqji->ijkl',fullD2,M,optimize="optimal")*2.

    #Final simplification
    #Works for residual construction
    if residual:
       A2  =-np.einsum('ijqp,klqp->ijkl',fullD2,M,optimize="optimal")*2.
       A2  -= A2.transpose(2,3,0,1)
    else:
    #Works for D2 update
       A2  =-np.einsum('ijqp,klqp->ijkl',fullD2,M,optimize="optimal")*2.
       A2  += A2.transpose(2,3,0,1)

    #Non-simplified D3 terms
    #A3  = np.einsum('ijprql,kprq->ijkl',fullD3,M,optimize="optimal")*3
    #A3 -= np.einsum('ijprqk,lprq->ijkl',fullD3,M,optimize="optimal")*3
    #A3 -= np.einsum('ijprql,pkrq->ijkl',fullD3,M,optimize="optimal")*3
    #A3 += np.einsum('ijprqk,plrq->ijkl',fullD3,M,optimize="optimal")*3
    #A3 -= np.einsum('jpqrlk,pqri->ijkl',fullD3,M,optimize="optimal")*3
    #A3 += np.einsum('jpqrlk,pqir->ijkl',fullD3,M,optimize="optimal")*3
    #A3 += np.einsum('ipqrlk,pqrj->ijkl',fullD3,M,optimize="optimal")*3
    #A3 -= np.einsum('ipqrlk,pqjr->ijkl',fullD3,M,optimize="optimal")*3

    #First simplification
    #A3  = np.einsum('ijprql,kprq->ijkl',fullD3,M,optimize="optimal")*6
    #A3 -= np.einsum('ijprqk,lprq->ijkl',fullD3,M,optimize="optimal")*6
    #A3 -= np.einsum('jpqrlk,pqri->ijkl',fullD3,M,optimize="optimal")*6
    #A3 += np.einsum('ipqrlk,pqrj->ijkl',fullD3,M,optimize="optimal")*6

    #Second simplification
    #A3  = np.einsum('ijprql,kprq->ijkl',fullD3,M,optimize="optimal")*6
    #A3 -= A3.transpose(0,1,3,2)
    #A3_ = np.einsum('ipqrlk,pqrj->ijkl',fullD3,M,optimize="optimal")*6
    #A3 += A3_ - A3_.transpose(1,0,2,3)

    #Works for residual construction
    if residual:
       A3  = np.einsum('ijprql,kprq->ijkl',fullD3,M,optimize="optimal")*6
       A3 -= A3.transpose(0,1,3,2)
       A3 += A3.transpose(3,2,0,1)
    else:
    #Works for D2 update
       A3  = np.einsum('ijprql,kprq->ijkl',fullD3,M,optimize="optimal")*6
       A3 -= A3.transpose(0,1,3,2)
       A3 -= A3.transpose(3,2,0,1)

    return A2 + A3

def get_acse_residual_full_no_D3(fullD1,fullD2,M,nelecas,ncore,reconstruction='V',residual=False):
    '''Calculates the residual/D2 update using the full 1- and 2-RDMs with sizes (2r,2r) and 
            (2r,2r,2r,2r) respectively where r is the number of spatial orbtials.

            Args:
                fullD1: 1RDM in the spin orbital basis
                fullD2: 2RDM in the spin orbital basis
                M: K2 for the residual or the residual for the D2 update
                nelecas: number of electrons in the active space. Tuple (Na,Nb)
                ncore: number of electrons in the core. Int
                reconstruction: Which reconstruction to use (V,NY)
                residual: Calculating residual? Boolean

            Returns:
                Residual or D2 update
            '''
    # J. Chem. Phys. 126, 184101 2007
    # TODO spin-separate
    # TODO parallelize
    r = fullD1.shape[0]//2
    d1wd1 = np.einsum('ik,jl->ijkl', fullD1, fullD1) - np.einsum('il,jk->ijkl', fullD1, fullD1)
    d1wd1 /= 2
    D2_c = fullD2-d1wd1
    tmp = (3*D2_c + d1wd1)
    nelecas = [nelecas[0]+ncore,nelecas[1]+ncore]

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

    if residual:
       A   = -np.einsum('ijqp,klqp->ijkl',fullD2,M,optimize="optimal")*2.
       A  -= A.transpose(2,3,0,1)
    else:
    #Works for D2 update
       A   = -np.einsum('ijqp,klqp->ijkl',fullD2,M,optimize="optimal")*2.
       A  += A.transpose(2,3,0,1)

    divisor = 1.5

    # 3RDM
    #A += np.einsum('ijpstl,pkst->ijkl',fullD3,M)*6
    #A1 = +np.einsum('kprq,ijrq,pl->ijkl',M,tmp,fullD1,optimize="optimal")/divisor
    #A2 = -np.einsum('kprq,ijlq,pr->ijkl',M,tmp,fullD1,optimize="optimal")/divisor
    #A3 = -np.einsum('kprq,ijrl,pq->ijkl',M,tmp,fullD1,optimize="optimal")/divisor
    #A4 = -np.einsum('kprq,iprq,jl->ijkl',M,tmp,fullD1,optimize="optimal")/divisor
    #A5 = +np.einsum('kprq,iplq,jr->ijkl',M,tmp,fullD1,optimize="optimal")/divisor
    #A6 = +np.einsum('kprq,iprl,jq->ijkl',M,tmp,fullD1,optimize="optimal")/divisor
    #A7 = +np.einsum('kprq,jprq,il->ijkl',M,tmp,fullD1,optimize="optimal")/divisor
    #A8 = -np.einsum('kprq,jplq,ir->ijkl',M,tmp,fullD1,optimize="optimal")/divisor
    #A9 = -np.einsum('kprq,jprl,iq->ijkl',M,tmp,fullD1,optimize="optimal")/divisor
    #A10 = A1 + A2 + A3 + A4 + A5 + A6+ A7 + A8 + A9



    #First simplification
    #A1 = +np.einsum('kprq,ijrq,pl->ijkl',M,tmp,fullD1,optimize="optimal")/divisor
    #A2 = -np.einsum('kprq,ijlq,pr->ijkl',M-M.transpose(0,1,3,2),tmp,fullD1,optimize="optimal")/divisor
    #A4 = -np.einsum('kprq,iprq,jl->ijkl',M,tmp,fullD1,optimize="optimal")/divisor
    #A5 = +np.einsum('kprq,iplq,jr->ijkl',M-M.transpose(0,1,3,2),tmp,fullD1,optimize="optimal")/divisor
    #A7 = +np.einsum('kprq,jprq,il->ijkl',M,tmp,fullD1,optimize="optimal")/divisor
    #A8 = -np.einsum('kprq,jplq,ir->ijkl',M-M.transpose(0,1,3,2),tmp,fullD1,optimize="optimal")/divisor
    #A10 = A1 + A2 + A4 + A5 - A5.transpose(1,0,2,3) + A7 + A8*0


    #Second simplification
    A1 = +np.einsum('kprq,ijrq,pl->ijkl',M,tmp,fullD1,optimize="optimal")
    A1 -= np.einsum('kprq,ijlq,pr->ijkl',M-M.transpose(0,1,3,2),tmp,fullD1,optimize="optimal")
    A2 = -np.einsum('kprq,iprq,jl->ijkl',M,tmp,fullD1,optimize="optimal")
    A2 += np.einsum('kprq,iplq,jr->ijkl',M-M.transpose(0,1,3,2),tmp,fullD1,optimize="optimal")
    A10 = A1 + A2 - A2.transpose(1,0,2,3) 
    A10 /= divisor


    if reconstruction == "NY":
        tmpB = np.zeros(A.shape)
        #Unsimplified version
        #tmpB = +np.einsum('kprq,a,iarq,jpal->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,iarq,pjal->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,jarq,ipal->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,parq,jial->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,parq,ijal->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,jarq,pial->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #                                                            
        #tmpB -= np.einsum('kprq,a,iaqr,jpal->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,iaqr,pjal->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,jaqr,ipal->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,paqr,jial->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,paqr,ijal->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,jaqr,pial->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #                                                            
        #tmpB -= np.einsum('kprq,a,ialq,jpar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,ialq,pjar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,jalq,ipar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,palq,jiar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,palq,ijar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,jalq,piar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #                                                            
        #tmpB -= np.einsum('kprq,a,iarl,jpaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,iarl,pjaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,jarl,ipaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,parl,jiaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,parl,ijaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,jarl,piaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #                                                            
        #tmpB += np.einsum('kprq,a,iaql,jpar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,iaql,pjar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,jaql,ipar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,paql,jiar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,paql,ijar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,jaql,piar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #                                                            
        #tmpB += np.einsum('kprq,a,ialr,jpaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,ialr,pjaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,jalr,ipaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB -= np.einsum('kprq,a,palr,jiaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,palr,ijaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)
        #tmpB += np.einsum('kprq,a,jalr,piaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)

        #First simplification
        #tmpB  = +np.einsum('kprq,a,iarq,jpal->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)*4
        #tmpB  -= np.einsum('kprq,a,jarq,ipal->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)*4
        #tmpB  -= np.einsum('kprq,a,parq,jial->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)*4
        #                                                              
        #tmpB  -= np.einsum('kprq,a,ialq,jpar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)*4
        #tmpB  += np.einsum('kprq,a,jalq,ipar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)*4
        #tmpB  += np.einsum('kprq,a,palq,jiar->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)*4
        #                                                            
        #tmpB  -= np.einsum('kprq,a,iarl,jpaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)*4
        #tmpB  += np.einsum('kprq,a,jarl,ipaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)*4
        #tmpB  += np.einsum('kprq,a,parl,jiaq->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)*4

        #Second simplification
        tmpB1  = +np.einsum('kprq,a,iarq,jpal->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)*4
        tmpB2  = -np.einsum('kprq,a,parq,jial->ijkl',M,HF_occs,D2_c,D2_c,optimize=True)*4
                                                                      
        tmpB1  -= np.einsum('kprq,a,ialq,jpar->ijkl',M-M.transpose(0,1,3,2),HF_occs,D2_c,D2_c,optimize=True)*4
        tmpB2  += np.einsum('kprq,a,palq,jiar->ijkl',M-M.transpose(0,1,3,2),HF_occs,D2_c,D2_c,optimize=True)*4

        tmpB = tmpB1 - tmpB1.transpose(1,0,2,3) + tmpB2
                                                                    
        A10 += tmpB

    if residual:
       #A3  = np.einsum('ijprql,kprq->ijkl',fullD3,M,optimize="optimal")*6
       A10 -= A10.transpose(0,1,3,2)
       A10 += A10.transpose(3,2,0,1)
    else:
    #Works for D2 update
       #A  = np.einsum('ijprql,kprq->ijkl',fullD3,M,optimize="optimal")*6
       A10 -= A10.transpose(0,1,3,2)
       A10 -= A10.transpose(3,2,0,1)
    
    A+=A10

    return A

def get_energy_blockD2(D2,K2):  # TODO factor of half in aa/bb part?
    E_aa = np.einsum('ijkl,ijkl',K2,D2[0])
    E_ab = np.einsum('ijkl,ijkl',K2,D2[1])*2
    E_bb = np.einsum('ijkl,ijkl',K2,D2[2])

    return E_aa + E_ab + E_bb

def get_acse_residual_blocks_no_D3(BlockedD1,BlockedD2,BlockedM,nelecas,ncore,reconstruction='V',A=True):
    '''Calculates the residual/D2 update using the blocked 1- and 2-RDMs with sizes (2,r,r) and 
            (3,r,r,r,r) respectively where r is the number of spatial orbtials.

            Args:
                fullD1: 1RDM spin blocked
                fullD2: 2RDM spin blocked
                M: K2 for the residual or the residual for the D2 update
                nelecas: number of electrons in the active space. Tuple (Na,Nb)
                ncore: number of electrons in the core. Int
                reconstruction: Which reconstruction to use (V,NY)
                residual: Calculating residual? Boolean

            Returns:
                Residual or D2 update
            '''
    # J. Chem. Phys. 126, 184101 2007
    # TODO spin-separate
    # TODO parallelize
    #A is to denote if it is creating A or the update to D2. Results in an index sign swap
    r = BlockedD1[0].shape[0]

    Da = BlockedD1[0]
    Db = BlockedD1[1]
    Daa = BlockedD2[0]
    Dab = BlockedD2[1]
    Dbb = BlockedD2[2]
    Maa = BlockedM[0]
    Mab = BlockedM[1]
    Mbb = BlockedM[2]

    d1wd1_aa = np.einsum('ik,jl->ijkl',Da,Da) - np.einsum('il,jk->ijkl',Da,Da)
    d1wd1_ab = np.einsum('ik,jl->ijkl',Da,Db) #- np.einsum('il,jk->ijkl',Da,Da)
    d1wd1_bb = np.einsum('ik,jl->ijkl',Db,Db) - np.einsum('il,jk->ijkl',Db,Db)

    D2c_aa = (Daa*2-d1wd1_aa)*3
    D2c_ab = (Dab*2-d1wd1_ab)*3
    D2c_bb = (Dbb*2-d1wd1_bb)*3

    tmp_aa = D2c_aa + d1wd1_aa
    tmp_ab = D2c_ab + d1wd1_ab
    tmp_bb = D2c_bb + d1wd1_bb

    D2c_aa /= 6
    D2c_ab /= 6
    D2c_bb /= 6
   
    #2RDM based terms
    A_aa  = np.einsum('ijkl,stkl->ijst',Daa,Maa,optimize="optimal")*2.
    A_ab  = np.einsum('ijkl,stkl->ijst',Dab,Mab,optimize="optimal")*4.
    A_bb  = np.einsum('ijkl,stkl->ijst',Dbb,Mbb,optimize="optimal")*2.
    if A:
        A_ab -= A_ab.transpose(2,3,0,1)
        A_aa -= A_aa.transpose(2,3,0,1)
        A_bb -= A_bb.transpose(2,3,0,1)
    else:
        A_ab += A_ab.transpose(2,3,0,1)
        A_aa += A_aa.transpose(2,3,0,1)
        A_bb += A_bb.transpose(2,3,0,1)
    
    #3RDM aa terms
    if A:
        tmp  = np.einsum('ijkl,pt,jskl->pist',tmp_aa,Da,Maa,optimize="optimal")/3
        tmp += np.einsum('ijkl,pt,jstk->pisl',tmp_aa,Da,Maa,optimize="optimal")/3
        tmp -= np.einsum('ijkl,pt,jskt->pisl',tmp_aa,Da,Maa,optimize="optimal")/3
        tmp -= np.einsum('ijkl,pt,sjtl->piks',tmp_ab,Da,Mab,optimize="optimal")/3*2 #abba,aa,baba
        tmp += np.einsum('ijkl,pt,sjkl->pits',tmp_ab,Da,Mab,optimize="optimal")/3*2 #abab,aa,baab
        tmp += tmp.transpose(3,2,0,1)
    else:
        tmp =  np.einsum('ijkl,pt,jstk->pisl',tmp_aa,Da,Maa,optimize="optimal")/3*2
        tmp += tmp.transpose(3,2,1,0)
        tmp += np.einsum('ijkl,pt,jskl->pist',tmp_aa,Da,Maa,optimize="optimal")/3*2
        tmp -= np.einsum('ijkl,pt,sjtl->piks',tmp_ab,Da,Mab,optimize="optimal")/3*4 #abba,aa,baba
        tmp += np.einsum('ijkl,pt,ijsl->pskt',tmp_ab,Da,Mab,optimize="optimal")/3*4 #abab,aa,abba
    A_aa += tmp - tmp.transpose(0,1,3,2) - tmp.transpose(1,0,2,3) + tmp.transpose(1,0,3,2)

    tmp = -np.einsum('ijkl,pt,ijts->spkl',tmp_aa,Da,Maa,optimize="optimal")/3
    tmp += np.einsum('ijkl,pt,ipst->jskl',tmp_aa,Db,Mab,optimize="optimal")/3*2 #aaaa,bb,baba
    if A:
        tmp -= np.einsum('ijkl,pt,pits->sjkl',tmp_aa,Da,Maa,optimize="optimal")/3
        tmp -= np.einsum('ijkl,pt,jpts->sikl',tmp_aa,Da,Maa,optimize="optimal")/3
    else:
        tmp -= np.einsum('ijkl,pt,pits->sjkl',tmp_aa,Da,Maa,optimize="optimal")/3*2
    tmp -= tmp.transpose(1,0,2,3)
    if A:
        A_aa += tmp - tmp.transpose(2,3,0,1)
    else:
        A_aa += tmp + tmp.transpose(2,3,0,1)


    #3RDM bb terms
    if A:
        tmp  = np.einsum('ijkl,pt,jskl->pist',tmp_bb,Db,Mbb,optimize="optimal")/3
        tmp += np.einsum('ijkl,pt,jstk->pisl',tmp_bb,Db,Mbb,optimize="optimal")/3
        tmp -= np.einsum('ijkl,pt,jskt->pisl',tmp_bb,Db,Mbb,optimize="optimal")/3
        tmp -= np.einsum('jilk,pt,jslt->piks',tmp_ab,Db,Mab,optimize="optimal")/3*2 #abba,aa,baba
        tmp += np.einsum('jilk,pt,jslk->pits',tmp_ab,Db,Mab,optimize="optimal")/3*2 #abab,aa,baab
        tmp += tmp.transpose(3,2,0,1)
    else:
        tmp =  np.einsum('ijkl,pt,jstk->pisl',tmp_bb,Db,Mbb,optimize="optimal")/3*2
        tmp += tmp.transpose(3,2,1,0)
        tmp += np.einsum('ijkl,pt,jskl->pist',tmp_bb,Db,Mbb,optimize="optimal")/3*2
        tmp -= np.einsum('jilk,pt,jslt->piks',tmp_ab,Db,Mab,optimize="optimal")/3*4 #abba,aa,baba
        tmp += np.einsum('jilk,pt,jils->pskt',tmp_ab,Db,Mab,optimize="optimal")/3*4 #abab,aa,abba
    A_bb += tmp - tmp.transpose(0,1,3,2) - tmp.transpose(1,0,2,3) + tmp.transpose(1,0,3,2)

    tmp = -np.einsum('ijkl,pt,ijts->spkl',tmp_bb,Db,Mbb,optimize="optimal")/3
    tmp += np.einsum('ijkl,pt,pits->jskl',tmp_bb,Da,Mab,optimize="optimal")/3*2 #aaaa,bb,baba
    if A:
        tmp -= np.einsum('ijkl,pt,pits->sjkl',tmp_bb,Db,Mbb,optimize="optimal")/3
        tmp -= np.einsum('ijkl,pt,jpts->sikl',tmp_bb,Db,Mbb,optimize="optimal")/3
    else:
        tmp -= np.einsum('ijkl,pt,pits->sjkl',tmp_bb,Db,Mbb,optimize="optimal")/3*2
    tmp -= tmp.transpose(1,0,2,3)
    if A:
        A_bb += tmp - tmp.transpose(2,3,0,1)
    else:
        A_bb += tmp + tmp.transpose(2,3,0,1)

    #3RDM ab terms
    tmp  = -np.einsum('ijkl,pt,sjtk->pisl',tmp_bb,Da,Mab,optimize="optimal")/3*2 #bbbb,aa,baba
    tmp  -= np.einsum('ijkl,pt,jskl->pits',tmp_bb,Da,Mbb,optimize="optimal")/3
    tmp  += np.einsum('ijkl,pt,iskl->jpst',tmp_aa,Db,Maa,optimize="optimal")/3 
    tmp  += np.einsum('ijkl,pt,iskt->jpls',tmp_aa,Db,Mab,optimize="optimal")/3*2 #baba,bb,abab
    tmp  += np.einsum('ijkl,pt,spkt->ijsl',tmp_ab,Db,Mab,optimize="optimal")/3*2 #abab,bb,baab
    tmp  -= np.einsum('ijkl,pt,sjkt->ipsl',tmp_ab,Db,Mab,optimize="optimal")/3*2 #baab,bb,baab
    tmp  -= np.einsum('ijkl,pt,spkl->ijst',tmp_ab,Db,Mab,optimize="optimal")/3*2 #abab,bb,baab
    tmp  += np.einsum('ijkl,pt,sjkl->ipst',tmp_ab,Db,Mab,optimize="optimal")/3*2 #baba,bb,baba
    tmp  -= np.einsum('ijkl,pt,pskl->ijts',tmp_ab,Da,Mab,optimize="optimal")/3*2 #abab,aa,abab
    tmp  += np.einsum('ijkl,pt,iskl->pjts',tmp_ab,Da,Mab,optimize="optimal")/3*2 #baab,aa,abab
    tmp  += np.einsum('ijkl,pt,pstl->ijks',tmp_ab,Da,Mab,optimize="optimal")/3*2 #abba,aa,abba
    tmp  -= np.einsum('ijkl,pt,istl->pjks',tmp_ab,Da,Mab,optimize="optimal")/3*2 #baba,aa,abba
    if A:
        tmp -= np.einsum('ijks,pt,plst->ijkl',tmp_ab,Db,Mbb,optimize="optimal")/3 #abba,bb,bbbb
        tmp += np.einsum('ijkt,ps,plst->ijkl',tmp_ab,Db,Mbb,optimize="optimal")/3 #abba,bb,bbbb
        tmp += np.einsum('ipks,jt,plst->ijkl',tmp_ab,Db,Mbb,optimize="optimal")/3 #baba,bb,bbbb
        tmp -= np.einsum('ipkt,js,plst->ijkl',tmp_ab,Db,Mbb,optimize="optimal")/3
        tmp -= np.einsum('ijsl,pt,pkst->ijkl',tmp_ab,Da,Maa,optimize="optimal")/3 #abab,aa,aaaa
        tmp += np.einsum('ijtl,ps,pkst->ijkl',tmp_ab,Da,Maa,optimize="optimal")/3 #abab,aa,aaaa
        tmp += np.einsum('pjsl,it,pkst->ijkl',tmp_ab,Da,Maa,optimize="optimal")/3 #baab,aa,aaaa
        tmp -= np.einsum('pjtl,is,pkst->ijkl',tmp_ab,Da,Maa,optimize="optimal")/3
        tmp -= tmp.transpose(2,3,0,1)
        A_ab += tmp 
    else:
        tmp -= np.einsum('ijkl,pt,pslt->ijks',tmp_ab,Db,Mbb,optimize="optimal")/3*2 #abba,bb,bbbb
        tmp -= np.einsum('ijkl,pt,jstl->ipks',tmp_ab,Db,Mbb,optimize="optimal")/3*2
        tmp -= np.einsum('ijkl,pt,pskt->ijsl',tmp_ab,Da,Maa,optimize="optimal")/3*2 #abab,aa,aaaa
        tmp -= np.einsum('pjtl,is,pkst->ijkl',tmp_ab,Da,Maa,optimize="optimal")/3*2
        tmp += tmp.transpose(2,3,0,1)
        A_ab += tmp

    #Building HF reference occupations
    nelecas = [nelecas[0]+ncore,nelecas[1]+ncore]
    HF_occs = [np.zeros(r),np.zeros(r)]
    for i in range(r):
        if i < nelecas[0]:
            HF_occs[0][i] = 1
        else:
            HF_occs[0][i] = -1
        if i < nelecas[1]:
            HF_occs[1][i] = 1
        else:
            HF_occs[1][i] = -1

    if reconstruction == "NY":
        #NY 3RDM aa terms
        if A:
            tmpB_   =  4*np.einsum('r,srtu,vwrx,wytu->svyx',HF_occs[0],D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB_  +=  4*np.einsum('r,srtu,vwrx,wyxt->svyu',HF_occs[0],D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB_  +=  4*np.einsum('r,srtu,vwrx,wyux->svyt',HF_occs[0],D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB_  -=  tmpB_.transpose(1,0,2,3)
            tmpB_  +=  4*np.einsum('r,srtu,vwrx,sytx->wvyu',HF_occs[0],D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB_  +=  4*np.einsum('r,srtu,vwrx,syxu->wvyt',HF_occs[0],D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB_  +=  4*np.einsum('r,srtu,vwrx,syut->wvyx',HF_occs[0],D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB_  -= tmpB_.transpose(0,1,3,2)
            tmpB    = tmpB_ - tmpB_.transpose(2,3,0,1)
        else:
            tmpB_  =  32*np.einsum('r,srtu,vwrx,sytx->wvyu',HF_occs[0],D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB_ +=  16*np.einsum('r,srtu,vwrx,syut->wvyx',HF_occs[0],D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB_ +=  64*np.einsum('r,srtu,vwrx,wyxt->svyu',HF_occs[0],D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB_ +=  32*np.einsum('r,srtu,vwrx,wvuy->sytx',HF_occs[0],D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB = tmpB_
        tmpB_   =  8*np.einsum('r,srtu,vwxr,ywtu->svyx',HF_occs[1],D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpB_  +=  8*np.einsum('r,srtu,vwxr,ywxu->vsyt',HF_occs[1],D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmp1    =  8*np.einsum('r,rsut,vwrx,ysxt->wvyu',HF_occs[0],D2c_ab,D2c_aa,Mab,optimize="optimal")
        tmp1   +=  8*np.einsum('r,rsut,vwrx,ysut->vwyx',HF_occs[0],D2c_ab,D2c_aa,Mab,optimize="optimal")
        tmp2    =  8*np.einsum('r,rsut,vwrx,vsyt->wyux',HF_occs[0],D2c_ab,D2c_aa,Mab,optimize="optimal")
        if A:
            tmpB_ += tmpB_.transpose(2,3,1,0)
            tmp2  -= tmp2.transpose(3,2,1,0)
            tmp1  += tmp1.transpose(2,3,1,0)
        else:
            tmpB_ -= tmpB_.transpose(2,3,1,0)
            tmp2  += tmp2.transpose(3,2,1,0)
            tmp1  -= tmp1.transpose(2,3,1,0)
        tmpB_ += tmp2
        tmpB_ -= tmpB_.transpose(0,1,3,2) - tmp1
        tmpB += tmpB_ + tmpB_.transpose(1,0,3,2)
        A_aa += tmpB


        #NY 3RDM bb terms
        tmpC_  = -8*np.einsum('r,srtu,vwxr,vyxu->swyt',HF_occs[1],D2c_bb,D2c_ab,Mab,optimize="optimal")
        tmpC_ += -8*np.einsum('r,srtu,vwxr,vyxt->wsyu',HF_occs[1],D2c_bb,D2c_ab,Mab,optimize="optimal")
        tmpC_ += -8*np.einsum('r,srtu,vwxr,vsxy->wyut',HF_occs[1],D2c_bb,D2c_ab,Mab,optimize="optimal")
        tmpC_ += -8*np.einsum('r,srtu,vwxr,vwxy->sytu',HF_occs[1],D2c_bb,D2c_ab,Mab,optimize="optimal")
        tmpB_  =  8*np.einsum('r,rsut,vwrx,vyut->swyx',HF_occs[0],D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpB_ +=  8*np.einsum('r,rsut,vwrx,vyux->wsyt',HF_occs[0],D2c_ab,D2c_ab,Mab,optimize="optimal")
        if A:
            tmpB_ += tmpB_.transpose(2,3,1,0)
            tmpC_ -= tmpC_.transpose(3,2,1,0)
        else:
            tmpB_ -= tmpB_.transpose(2,3,1,0)
            tmpC_ += tmpC_.transpose(3,2,1,0)
        tmpC_ += tmpB_ - tmpB_.transpose(0,1,3,2)
        tmpB  = tmpC_ + tmpC_.transpose(1,0,3,2)

        if A:
            tmpB_   =  4*np.einsum('r,srtu,vwrx,wytu->svyx',HF_occs[1],D2c_bb,D2c_bb,Mbb,optimize="optimal")
            tmpB_  +=  4*np.einsum('r,srtu,vwrx,wyxt->svyu',HF_occs[1],D2c_bb,D2c_bb,Mbb,optimize="optimal")
            tmpB_  +=  4*np.einsum('r,srtu,vwrx,wyux->svyt',HF_occs[1],D2c_bb,D2c_bb,Mbb,optimize="optimal")
            tmpB_  -=  tmpB_.transpose(1,0,2,3)
            tmpB_  +=  4*np.einsum('r,srtu,vwrx,sytx->wvyu',HF_occs[1],D2c_bb,D2c_bb,Mbb,optimize="optimal")
            tmpB_  +=  4*np.einsum('r,srtu,vwrx,syxu->wvyt',HF_occs[1],D2c_bb,D2c_bb,Mbb,optimize="optimal")
            tmpB_  +=  4*np.einsum('r,srtu,vwrx,syut->wvyx',HF_occs[1],D2c_bb,D2c_bb,Mbb,optimize="optimal")
            tmpB_  -= tmpB_.transpose(0,1,3,2)
            tmpB   += tmpB_ - tmpB_.transpose(2,3,0,1)
        else:
            tmpB_  =  32*np.einsum('r,srtu,vwrx,sytx->wvyu',HF_occs[1],D2c_bb,D2c_bb,Mbb,optimize="optimal")
            tmpB_ +=  16*np.einsum('r,srtu,vwrx,syut->wvyx',HF_occs[1],D2c_bb,D2c_bb,Mbb,optimize="optimal")
            tmpB_ +=  64*np.einsum('r,srtu,vwrx,wyxt->svyu',HF_occs[1],D2c_bb,D2c_bb,Mbb,optimize="optimal")
            tmpB_ +=  32*np.einsum('r,srtu,vwrx,wvuy->sytx',HF_occs[1],D2c_bb,D2c_bb,Mbb,optimize="optimal")
            tmpB  += tmpB_
        A_bb += tmpB

        #NY 3RDM ab terms
        tmpB  = -4*np.einsum('r,srtu,vwrx,vytu->swyx',HF_occs[0],D2c_aa,D2c_ab,Maa,optimize="optimal")
        tmpB -=  4*np.einsum('r,srtu,vwrx,syut->vwyx',HF_occs[0],D2c_aa,D2c_ab,Maa,optimize="optimal")
        if A:
            tmpB +=  4*np.einsum('r,rsut,vwrx,wyxu->vsyt',HF_occs[0],D2c_ab,D2c_aa,Maa,optimize="optimal")
            tmpB +=  4*np.einsum('r,rsut,vwrx,vyux->wsyt',HF_occs[0],D2c_ab,D2c_aa,Maa,optimize="optimal")
            tmpB -=  4*np.einsum('r,srtu,vwxr,vytx->swyu',HF_occs[1],D2c_ab,D2c_ab,Maa,optimize="optimal")
            tmpB +=  4*np.einsum('r,srtu,vwxr,vyxt->swyu',HF_occs[1],D2c_ab,D2c_ab,Maa,optimize="optimal")
            tmpB +=  4*np.einsum('r,srtu,vwxr,sytx->vwyu',HF_occs[1],D2c_ab,D2c_ab,Maa,optimize="optimal")
            tmpB -=  4*np.einsum('r,srtu,vwxr,syxt->vwyu',HF_occs[1],D2c_ab,D2c_ab,Maa,optimize="optimal")
        else:
            tmpB +=  8*np.einsum('r,rsut,vwrx,wyxu->vsyt',HF_occs[0],D2c_ab,D2c_aa,Maa,optimize="optimal")
            tmpB -=  8*np.einsum('r,srtu,vwxr,vytx->swyu',HF_occs[1],D2c_ab,D2c_ab,Maa,optimize="optimal")
            tmpB -=  8*np.einsum('r,srtu,vwxr,syxt->vwyu',HF_occs[1],D2c_ab,D2c_ab,Maa,optimize="optimal")
        tmpB -=  8*np.einsum('r,rsut,wvrx,yvut->wsyx',HF_occs[0],D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpB +=  8*np.einsum('r,rsut,wvrx,yvux->wsyt',HF_occs[0],D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpB +=  8*np.einsum('r,rsut,wvrx,ysut->wvyx',HF_occs[0],D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpB -=  8*np.einsum('r,rsut,wvrx,ysux->wvyt',HF_occs[0],D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpB -=  8*np.einsum('r,srtu,vwxr,ywxt->vsyu',HF_occs[1],D2c_bb,D2c_ab,Mab,optimize="optimal")
        tmpB -=  8*np.einsum('r,srtu,vwxr,ysxu->vwyt',HF_occs[1],D2c_bb,D2c_ab,Mab,optimize="optimal")
        tmpB +=  8*np.einsum('r,srtu,vwrx,ywtx->svyu',HF_occs[1],D2c_ab,D2c_bb,Mab,optimize="optimal")
        tmpB +=  8*np.einsum('r,srtu,vwrx,yvtu->swyx',HF_occs[1],D2c_ab,D2c_bb,Mab,optimize="optimal")
        tmpC_ = tmpB.copy()

        tmpE  = -8*np.einsum('r,srtu,vwrx,vytx->swuy',HF_occs[0],D2c_aa,D2c_ab,Mab,optimize="optimal")
        tmpE += -8*np.einsum('r,srtu,vwrx,syux->vwty',HF_occs[0],D2c_aa,D2c_ab,Mab,optimize="optimal")
        tmpE += -8*np.einsum('r,rsut,vwrx,wyut->vsxy',HF_occs[0],D2c_ab,D2c_aa,Mab,optimize="optimal")
        tmpE +=  8*np.einsum('r,rsut,vwrx,wyxt->vsuy',HF_occs[0],D2c_ab,D2c_aa,Mab,optimize="optimal")

        if A:
            tmpE -=  4*np.einsum('r,rsut,vwrx,wytx->vsuy',HF_occs[0],D2c_ab,D2c_ab,Mbb,optimize="optimal")
            tmpE +=  4*np.einsum('r,rsut,vwrx,wyxt->vsuy',HF_occs[0],D2c_ab,D2c_ab,Mbb,optimize="optimal")
            tmpE +=  4*np.einsum('r,rsut,vwrx,sytx->vwuy',HF_occs[0],D2c_ab,D2c_ab,Mbb,optimize="optimal")
            tmpE -=  4*np.einsum('r,rsut,vwrx,syxt->vwuy',HF_occs[0],D2c_ab,D2c_ab,Mbb,optimize="optimal")
            tmpE +=  4*np.einsum('r,srtu,vwrx,wyxu->svty',HF_occs[1],D2c_ab,D2c_bb,Mbb,optimize="optimal")
            tmpE +=  4*np.einsum('r,srtu,vwrx,vyux->swty',HF_occs[1],D2c_ab,D2c_bb,Mbb,optimize="optimal")
        else:
            tmpE +=  8*np.einsum('r,rsut,vwrx,wyxt->vsuy',HF_occs[0],D2c_ab,D2c_ab,Mbb,optimize="optimal")
            tmpE +=  8*np.einsum('r,rsut,vwrx,sytx->vwuy',HF_occs[0],D2c_ab,D2c_ab,Mbb,optimize="optimal")
            tmpE +=  8*np.einsum('r,srtu,vwrx,wyxu->svty',HF_occs[1],D2c_ab,D2c_bb,Mbb,optimize="optimal")

        tmpE -=  4*np.einsum('r,srtu,vwxr,wytu->vsxy',HF_occs[1],D2c_bb,D2c_ab,Mbb,optimize="optimal")
        tmpE -=  4*np.einsum('r,srtu,vwxr,syut->vwxy',HF_occs[1],D2c_bb,D2c_ab,Mbb,optimize="optimal")

        tmpE -=  8*np.einsum('r,srut,vwxr,vyut->swxy',HF_occs[1],D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpE +=  8*np.einsum('r,srut,vwxr,vyxt->swuy',HF_occs[1],D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpE +=  8*np.einsum('r,srut,vwxr,syut->vwxy',HF_occs[1],D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpE -=  8*np.einsum('r,srut,vwxr,syxt->vwuy',HF_occs[1],D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpD_ = tmpE.copy()

        if A:
            A_ab += tmpC_ - tmpC_.transpose(2,3,0,1)
            A_ab += tmpD_ - tmpD_.transpose(2,3,0,1)
        else:
            A_ab += tmpC_ + tmpC_.transpose(2,3,0,1)
            A_ab += tmpD_ + tmpD_.transpose(2,3,0,1)
    return [A_aa,A_ab,A_bb]


def get_acse_residual_blocks_no_D3_singlet(BlockedD1,BlockedD2,BlockedM,nelecas,ncore,reconstruction='V',A=True):
    '''Calculates the residual/D2 update using the blocked 1- and 2-RDMs with sizes (2,r,r) and 
            (3,r,r,r,r) respectively where r is the number of spatial orbtials. Assumes a 
            singlet

            Args:
                fullD1: 1RDM spin blocked
                fullD2: 2RDM spin blocked
                M: K2 for the residual or the residual for the D2 update
                nelecas: number of electrons in the active space. Tuple (Na,Nb)
                ncore: number of electrons in the core. Int
                reconstruction: Which reconstruction to use (V,NY)
                residual: Calculating residual? Boolean

            Returns:
                Residual or D2 update
            '''
    # J. Chem. Phys. 126, 184101 2007
    # TODO spin-separate
    # TODO parallelize
    #A is to denote if it is creating A or the update to D2. Results in an index sign swap
    r = BlockedD1[0].shape[0]

    Da = BlockedD1[0]
    Daa = BlockedD2[0]
    Dab = BlockedD2[1]
    Maa = BlockedM[0]
    Mab = BlockedM[1]

    d1wd1_aa = np.einsum('ik,jl->ijkl',Da,Da) - np.einsum('il,jk->ijkl',Da,Da)
    d1wd1_ab = np.einsum('ik,jl->ijkl',Da,Da) #- np.einsum('il,jk->ijkl',Da,Da)

    D2c_aa = (Daa*2-d1wd1_aa)*3
    D2c_ab = (Dab*2-d1wd1_ab)*3

    tmp_aa = D2c_aa + d1wd1_aa
    tmp_ab = D2c_ab + d1wd1_ab

    D2c_aa /= 6
    D2c_ab /= 6
   

    A_aa  = np.einsum('ijkl,stkl->ijst',Daa,Maa,optimize="optimal")*2.
    A_ab  = np.einsum('ijkl,stkl->ijst',Dab,Mab,optimize="optimal")*4.
    if A:
        A_ab -= A_ab.transpose(2,3,0,1)
        A_aa -= A_aa.transpose(2,3,0,1)
    else:
        A_ab += A_ab.transpose(2,3,0,1)
        A_aa += A_aa.transpose(2,3,0,1)
    
    #A_aa
    if A:
        tmp  = np.einsum('ijkl,pt,jskl->pist',tmp_aa,Da,Maa,optimize="optimal")/3
        tmp += np.einsum('ijkl,pt,jstk->pisl',tmp_aa,Da,Maa,optimize="optimal")/3
        tmp -= np.einsum('ijkl,pt,jskt->pisl',tmp_aa,Da,Maa,optimize="optimal")/3
        tmp -= np.einsum('ijkl,pt,sjtl->piks',tmp_ab,Da,Mab,optimize="optimal")/3*2 #abba,aa,baba
        tmp += np.einsum('ijkl,pt,sjkl->pits',tmp_ab,Da,Mab,optimize="optimal")/3*2 #abab,aa,baab
        tmp += tmp.transpose(3,2,0,1)
    else:
        tmp =  np.einsum('ijkl,pt,jstk->pisl',tmp_aa,Da,Maa,optimize="optimal")/3*2
        tmp += tmp.transpose(3,2,1,0)
        tmp += np.einsum('ijkl,pt,jskl->pist',tmp_aa,Da,Maa,optimize="optimal")/3*2
        tmp -= np.einsum('ijkl,pt,sjtl->piks',tmp_ab,Da,Mab,optimize="optimal")/3*4 #abba,aa,baba
        tmp += np.einsum('ijkl,pt,ijsl->pskt',tmp_ab,Da,Mab,optimize="optimal")/3*4 #abab,aa,abba
    A_aa += tmp - tmp.transpose(0,1,3,2) - tmp.transpose(1,0,2,3) + tmp.transpose(1,0,3,2)

    tmp = -np.einsum('ijkl,pt,ijts->spkl',tmp_aa,Da,Maa,optimize="optimal")/3
    tmp += np.einsum('ijkl,pt,ipst->jskl',tmp_aa,Da,Mab,optimize="optimal")/3*2 #aaaa,bb,baba
    if A:
        tmp -= np.einsum('ijkl,pt,pits->sjkl',tmp_aa,Da,Maa,optimize="optimal")/3
        tmp -= np.einsum('ijkl,pt,jpts->sikl',tmp_aa,Da,Maa,optimize="optimal")/3
    else:
        tmp -= np.einsum('ijkl,pt,pits->sjkl',tmp_aa,Da,Maa,optimize="optimal")/3*2
    tmp -= tmp.transpose(1,0,2,3)
    if A:
        A_aa += tmp - tmp.transpose(2,3,0,1)
    else:
        A_aa += tmp + tmp.transpose(2,3,0,1)

    #A_ab
    tmp   = -np.einsum('ijkl,pt,sjtk->pisl',tmp_aa,Da,Mab,optimize="optimal")/3*2 #bbbb,aa,baba
    tmp  -= np.einsum('ijkl,pt,jskl->pits',tmp_aa,Da,Maa,optimize="optimal")/3
    tmp  += np.einsum('ijkl,pt,spkt->ijsl',tmp_ab,Da,Mab,optimize="optimal")/3*2 #abab,bb,baab
    tmp  -= np.einsum('ijkl,pt,sjkt->ipsl',tmp_ab,Da,Mab,optimize="optimal")/3*2 #baab,bb,baab
    tmp  -= np.einsum('ijkl,pt,spkl->ijst',tmp_ab,Da,Mab,optimize="optimal")/3*2 #abab,bb,baab
    tmp  += np.einsum('ijkl,pt,sjkl->ipst',tmp_ab,Da,Mab,optimize="optimal")/3*2 #baba,bb,baba
    tmp += tmp.transpose(1,0,3,2)

    if A:
        tmp1 = -np.einsum('ijks,pt,plst->ijkl',tmp_ab,Da,Maa,optimize="optimal")/3 #abba,bb,bbbb
        tmp1 += np.einsum('ijkt,ps,plst->ijkl',tmp_ab,Da,Maa,optimize="optimal")/3 #abba,bb,bbbb
        tmp1 += np.einsum('ipks,jt,plst->ijkl',tmp_ab,Da,Maa,optimize="optimal")/3 #baba,bb,bbbb
        tmp1 -= np.einsum('ipkt,js,plst->ijkl',tmp_ab,Da,Maa,optimize="optimal")/3
        tmp1 += tmp1.transpose(1,0,3,2)
        tmp += tmp1
        tmp -= tmp.transpose(2,3,0,1)
        A_ab += tmp 
    else:
        tmp1 = -np.einsum('ijkl,pt,pslt->ijks',tmp_ab,Da,Maa,optimize="optimal")/3*2 #abba,bb,bbbb
        tmp1 -= np.einsum('ijkl,pt,jstl->ipks',tmp_ab,Da,Maa,optimize="optimal")/3*2
        tmp1 += tmp1.transpose(1,0,3,2)
        tmp += tmp1
        tmp += tmp.transpose(2,3,0,1)
        A_ab += tmp


    nelecas = nelecas[0]+ncore

    HF_occs = np.zeros(r)
    for i in range(r):
        if i < nelecas:
            HF_occs[i] = 1
        else:
            HF_occs[i] = -1

    if reconstruction == "NY":

        if A:
            tmpB   =  4*np.einsum('r,srtu,vwrx,wytu->svyx',HF_occs,D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB  +=  4*np.einsum('r,srtu,vwrx,wyxt->svyu',HF_occs,D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB  +=  4*np.einsum('r,srtu,vwrx,wyux->svyt',HF_occs,D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB  -=  tmpB.transpose(1,0,2,3)
            tmpB  +=  4*np.einsum('r,srtu,vwrx,sytx->wvyu',HF_occs,D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB  +=  4*np.einsum('r,srtu,vwrx,syxu->wvyt',HF_occs,D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB  +=  4*np.einsum('r,srtu,vwrx,syut->wvyx',HF_occs,D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB  -= tmpB.transpose(0,1,3,2)
            tmpB  -= tmpB.transpose(2,3,0,1)
        else:
            tmpB  =  32*np.einsum('r,srtu,vwrx,sytx->wvyu',HF_occs,D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB +=  16*np.einsum('r,srtu,vwrx,syut->wvyx',HF_occs,D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB +=  64*np.einsum('r,srtu,vwrx,wyxt->svyu',HF_occs,D2c_aa,D2c_aa,Maa,optimize="optimal")
            tmpB +=  32*np.einsum('r,srtu,vwrx,wvuy->sytx',HF_occs,D2c_aa,D2c_aa,Maa,optimize="optimal")

        tmpB_   =  8*np.einsum('r,srtu,vwxr,ywtu->svyx',HF_occs,D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpB_  +=  8*np.einsum('r,srtu,vwxr,ywxu->vsyt',HF_occs,D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmp1    =  8*np.einsum('r,rsut,vwrx,ysxt->wvyu',HF_occs,D2c_ab,D2c_aa,Mab,optimize="optimal")
        tmp1   +=  8*np.einsum('r,rsut,vwrx,ysut->vwyx',HF_occs,D2c_ab,D2c_aa,Mab,optimize="optimal")
        tmp2    =  8*np.einsum('r,rsut,vwrx,vsyt->wyux',HF_occs,D2c_ab,D2c_aa,Mab,optimize="optimal")
        if A:
            tmpB_ += tmpB_.transpose(2,3,1,0)
            tmp2  -= tmp2.transpose(3,2,1,0)
            tmp1  += tmp1.transpose(2,3,1,0)
        else:
            tmpB_ -= tmpB_.transpose(2,3,1,0)
            tmp2  += tmp2.transpose(3,2,1,0)
            tmp1  -= tmp1.transpose(2,3,1,0)
        tmpB_ += tmp2
        tmpB_ -= tmpB_.transpose(0,1,3,2) - tmp1
        tmpB += tmpB_ + tmpB_.transpose(1,0,3,2)
        A_aa += tmpB

        
        tmpB  = -4*np.einsum('r,srtu,vwrx,vytu->swyx',HF_occs,D2c_aa,D2c_ab,Maa,optimize="optimal")
        tmpB -=  4*np.einsum('r,srtu,vwrx,syut->vwyx',HF_occs,D2c_aa,D2c_ab,Maa,optimize="optimal")

        if A:
            tmpB +=  4*np.einsum('r,rsut,vwrx,wyxu->vsyt',HF_occs,D2c_ab,D2c_aa,Maa,optimize="optimal")
            tmpB +=  4*np.einsum('r,rsut,vwrx,vyux->wsyt',HF_occs,D2c_ab,D2c_aa,Maa,optimize="optimal")
            tmpB -=  4*np.einsum('r,srtu,vwxr,vytx->swyu',HF_occs,D2c_ab,D2c_ab,Maa,optimize="optimal")
            tmpB +=  4*np.einsum('r,srtu,vwxr,vyxt->swyu',HF_occs,D2c_ab,D2c_ab,Maa,optimize="optimal")
            tmpB +=  4*np.einsum('r,srtu,vwxr,sytx->vwyu',HF_occs,D2c_ab,D2c_ab,Maa,optimize="optimal")
            tmpB -=  4*np.einsum('r,srtu,vwxr,syxt->vwyu',HF_occs,D2c_ab,D2c_ab,Maa,optimize="optimal")
        else:
            tmpB +=  8*np.einsum('r,srtu,wvxr,wyxu->vsyt',HF_occs,D2c_ab,D2c_aa,Maa,optimize="optimal")
            tmpB -=  8*np.einsum('r,srtu,vwxr,vytx->swyu',HF_occs,D2c_ab,D2c_ab,Maa,optimize="optimal")
            tmpB -=  8*np.einsum('r,srtu,vwxr,syxt->vwyu',HF_occs,D2c_ab,D2c_ab,Maa,optimize="optimal")

        tmpB -=  8*np.einsum('r,rsut,wvrx,yvut->wsyx',HF_occs,D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpB +=  8*np.einsum('r,rsut,wvrx,yvux->wsyt',HF_occs,D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpB +=  8*np.einsum('r,rsut,wvrx,ysut->wvyx',HF_occs,D2c_ab,D2c_ab,Mab,optimize="optimal")
        tmpB -=  8*np.einsum('r,rsut,wvrx,ysux->wvyt',HF_occs,D2c_ab,D2c_ab,Mab,optimize="optimal")

        tmpB -=  8*np.einsum('r,srtu,vwxr,ywxt->vsyu',HF_occs,D2c_aa,D2c_ab,Mab,optimize="optimal")
        tmpB -=  8*np.einsum('r,srtu,vwxr,ysxu->vwyt',HF_occs,D2c_aa,D2c_ab,Mab,optimize="optimal")
        tmpB +=  8*np.einsum('r,srtu,vwrx,ywtx->svyu',HF_occs,D2c_ab,D2c_aa,Mab,optimize="optimal")
        tmpB +=  8*np.einsum('r,srtu,vwrx,yvtu->swyx',HF_occs,D2c_ab,D2c_aa,Mab,optimize="optimal")
        tmpB += tmpB.transpose(1,0,3,2)

        if A:
            A_ab += tmpB - tmpB.transpose(2,3,0,1)
        else:
            A_ab += tmpB + tmpB.transpose(2,3,0,1)
    return [A_aa,A_ab,A_aa]
