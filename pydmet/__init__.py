import pyscf

import pydmet
from pydmet import vcor_fitting
from pydmet import solver

def RHF(m, solver=None, imp_lo_idx_list=None, 
           coeff_ao_lo=None,
           is_mu_fitting : bool = True,
           is_vcor_fitting : bool = True):

    from pydmet.dmet import DMETwithRHF

    dmet_obj = DMETwithRHF(m)
    dmet_obj.solver = solver
    dmet_obj.imp_lo_idx_list = imp_lo_idx_list
    dmet_obj.coeff_ao_lo = coeff_ao_lo
    dmet_obj.mu_fitting = is_mu_fitting
    dmet_obj.vcor_fitting = is_vcor_fitting
    
    return dmet_obj