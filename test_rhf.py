import os, sys
import numpy, scipy

import pyscf
from pyscf import lo, cc, mp

import pydmet

from pydmet.tools import mol_lo_tools
from pydmet import solver
import time

TOL = os.environ.get("TOL", 1e-7)

# Take care that the finite difference calculation of dn_dmu is super-sensitive
# to the convergence threshold of the HF solver, please make sure that the HF
# solver is converged to a very high accuracy.

def build(basis="sto3g"):
    import pyscf
    from   pyscf import lo
    time_start = time.time()
    mol = pyscf.gto.Mole()
    mol.build(
        atom = """
 C                 -4.41256978   -3.19242054    0.09811179
 C                 -3.01081978   -3.19242054    0.09811179
 C                 -2.58682178   -1.82162454    0.09811179
 C                 -3.74387878   -1.03017154    0.09800779
 N                 -4.85229678   -1.87191254    0.09810279
 H                 -5.11415778   -4.02565354    0.09820979
 H                 -2.36644678   -4.06512654    0.09813679
 H                 -1.56225878   -1.46513254    0.09808679
 H                 -3.85280078    0.05360046    0.09786079
 C                 -6.25671580   -1.43773849    0.09777869
 H                 -6.33857447   -0.49457463    0.59641792
 H                 -6.59919305   -1.33872060   -0.91108443
 H                 -6.85464366   -2.16388897    0.60776669
        """,
        basis = basis,
        verbose=0,
        charge=0
    )

    frag_atms_list   = [[0,1,5,6],[2,3,7,8],[4,9,10,11,12]]
    #frag_atms_list   = [[0,1],[2,3],[4,5]]


    mf = pyscf.scf.RHF(mol)
    mf.verbose = 4
    mf.max_cycle = 100
    mf.conv_tol  = TOL
    mf.conv_tol_grad = TOL
    mf.kernel()
    time_end = time.time()
    return mol, mf, frag_atms_list, time_end-time_start

def build_lo(mf, basis="sto3g"):
    coeff_ao_lo = None
    pm = lo.PM(mf.mol, mf.mo_coeff)
    pm.conv_tol = TOL
    coeff_ao_lo = pm.kernel()
    return coeff_ao_lo

def test_dmet_rhf_dn_dmu(basis="sto3g"):
    mol, mf, imp_atms_list, rhf_time = build(basis)
    dmet_time_start = time.time()
    ovlp_ao = mf.get_ovlp()
    coeff_ao_lo = build_lo(mf, basis)

    hcore_ao    = mf.get_hcore()
    ovlp_ao     = mf.get_ovlp()
    dm_ao       = mf.make_rdm1()
    fock_ao     = mf.get_fock(h1e=hcore_ao, dm=dm_ao)
    ene_hf_ref  = mf.energy_elec()[0]

    imp_lo_idx_list = mol_lo_tools.partition_lo_to_imps(
        imp_atms_list, mol=mol, coeff_ao_lo=coeff_ao_lo,
        min_weight=0.8
    )

    rhf_solver = solver.RCCSD()
    rhf_solver.max_cycle = 50
    rhf_solver.conv_tol      = TOL
    rhf_solver.conv_tol_grad = TOL
    rhf_solver.verbose = 0

    dmet_obj = pydmet.RHF(
        mf, solver=rhf_solver, 
        coeff_ao_lo=coeff_ao_lo,
        imp_lo_idx_list=imp_lo_idx_list,
        is_mu_fitting=True, 
        is_vcor_fitting=True
        )
    dmet_obj.verbose   = 0
    dmet_obj._hcore_ao = hcore_ao
    dmet_obj._ovlp_ao  = ovlp_ao
    dmet_obj._fock_ao  = fock_ao
    dmet_obj.build()
    dmet_obj.dump_flags()

    energy_elec, nelec_tot, dn_dmu, dm_hl_ao, dm_hl_lo = dmet_obj.kernel(mu0=0)
    energy_nuc = mol.energy_nuc()
    dmet_time_end = time.time()
    ccsd_time_start = time.time()
    refcc = cc.RCCSD(mf)
    refcc.verbose = 4
    refcc.kernel()
    e_ccsd = refcc.e_corr
    e_ccsd_t = e_ccsd + refcc.ccsd_t()
    ccsd_time_end = time.time()
    mp2_time_start = time.time()
    refmp2 = mp.MP2(mf)
    refmp2.verbose = 4
    refmp2.kernel()
    e_mp2 = refmp2.e_corr
    mp2_time_end = time.time()
    print(f'RHF energy     = {ene_hf_ref + energy_nuc:12.6f} Hartree')
    print(f'DMET energy    = {energy_elec + energy_nuc:12.6f} Hartree')
    print(f'MP2 energy     = {ene_hf_ref + e_mp2 + energy_nuc:12.6f} Hartree')
    print(f'CCSD(T) energy = {ene_hf_ref + e_ccsd_t + energy_nuc:12.6f} Hartree')
    print(f'RHF  cost   : {rhf_time:10.2f}s')
    print(f'DMET cost   : {rhf_time + dmet_time_end - dmet_time_start:10.2f}s')
    print(f'MP2 cost    : {rhf_time + mp2_time_end - mp2_time_start:10.2f}s')
    print(f'CCSD(T) cost: {rhf_time + ccsd_time_end - ccsd_time_start:10.2f}s')

test_dmet_rhf_dn_dmu(basis='sto3g')