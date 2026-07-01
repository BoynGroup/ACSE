import numpy as np
#import openfermion.linalg
#from openfermion.linalg import wedge
import math, itertools
from functools import wraps

def make_full_rdms_(casd1a, casd1b, casdm2, ncore, ncas, nmo, flg='aa'):
    '''
    flg : either 'aa' or 'ab'; use 'aa' for bb block
    '''
    
    nocc = ncas + ncore
    

    dm1a = np.zeros((nmo,nmo))
    idx = np.arange(ncore)
    dm1a[idx,idx] = 1
    dm1a[ncore:nocc,ncore:nocc] = casd1a
    
    dm1b = np.zeros((nmo,nmo))
    idx = np.arange(ncore)
    dm1b[idx,idx] = 1
    dm1b[ncore:nocc,ncore:nocc] = casd1b

    dm2 = np.zeros((nmo,nmo,nmo,nmo))
    if flg == 'aa':
        dm2 = np.einsum('ik,jl->ijkl',dm1a,dm1a) - np.einsum('jk,il->ijkl',dm1a,dm1a)
    elif flg == 'ab':
        dm2 = np.einsum('ik,jl->ijkl',dm1a,dm1b)
    dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2
    
    #for i in range(ncore):
    #    for j in range(ncore):
    #        dm2[i,j,i,j] += 2
    #        if flg == 'aa':
    #            dm2[i,j,j,i] += -2
    #        if flg == 'ab':
    #            dm2[i,j,j,i] += -1

    #    dm2[i,ncore:nocc,i,ncore:nocc] = dm2[ncore:nocc,i,ncore:nocc,i] =  (casd1a + casd1b)/2
    #    dm2[i,ncore:nocc,ncore:nocc,i] = -casd1a/2 # are these a/b in the correct order?
    #    dm2[ncore:nocc,i,i,ncore:nocc] = -casd1b/2
           
    return dm1a, dm1b, dm2


def make_full_rdms(mc):
    '''
    args : active space rdms
    returns : (d1a, d1b), (d2aa, d2ab, d2bb) in full MO space
    '''

    (da, db), (daa, dab, dbb) = mc.fcisolver.make_rdm12s(mc.ci, mc.ncas, mc.nelecas)
    #import numpy as np
    #from pyscf import fci
    #r = 6
    #dm1, dm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
    #from pyscf.fci.addons import civec_spinless_repr
    #ci_spinless = civec_spinless_repr([mc.ci,], r, [mc.nelecas,])
    #dm1, dm2, _ = fci.direct_spin1.make_rdm123(ci_spinless, 6*2, (mc.nelecas[0] + mc.nelecas[1],0))
    #print(dm2.shape)
    #dm2 = dm2.swapaxes(1,2)
    #print(np.linalg.norm(dm2-dm2.transpose(1,0,3,2)))
    #print(np.linalg.norm(dm2+dm2.transpose(1,0,2,3)))
    #exit()
    #print(da)
    #print(db)

    daa = daa.swapaxes(1,2)
    dab = dab.swapaxes(1,2)
    dbb = dbb.swapaxes(1,2)

    d1a, d1a, d2aa = make_full_rdms_(da, da, daa, mc.ncore, mc.ncas, mc.mol.nao_nr(), flg='aa') 
    d1b, d1b, d2bb = make_full_rdms_(db, db, dbb, mc.ncore, mc.ncas, mc.mol.nao_nr(), flg='aa') 
    d1a, d1b, d2ab = make_full_rdms_(da, db, dab, mc.ncore, mc.ncas, mc.mol.nao_nr(), flg='ab') 

    return [[d1a, d1b], [d2aa, d2ab, d2bb]]

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        start_time = time()
        result = f(*args, **kw)
        end_time = time()
        print('func:%r kw:[%r] took: %2.4f sec' % \
                (f.__name__, kw, end_time-start_time))
        return result
    return wrap
