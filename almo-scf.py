import numpy
from scipy.linalg import eigh
import pyscf, os

TOL = os.environ.get("TOL", 1e-8)

# Take care that the finite difference calculation of dn_dmu is super-sensitive
# to the convergence threshold of the HF solver, please make sure that the HF
# solver is converged to a very high accuracy.

def build(basis="sto3g"):

    mol = pyscf.gto.Mole()
    mol.build(
        atom = """
 C                 -3.08628339    1.03982299    0.00000000
 H                 -2.72962896    0.03101299    0.00000000
 H                 -2.72961055    1.54422118    0.87365150
 H                 -2.72961055    1.54422118   -0.87365150
 H                 -4.15628339    1.03983618    0.00000000
 C                  0.07743363   -0.73008848    0.00000000
 H                  0.43408806   -1.73889849    0.00000000
 H                  0.43410647   -0.22569029    0.87365150
 H                  0.43410647   -0.22569029   -0.87365150
 H                 -0.99256637   -0.73007530    0.00000000
        """,
        basis = basis,
        verbose=0
    )

    frag_atms_list   = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    
    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    mf.max_cycle = 100
    mf.conv_tol  = TOL
    mf.conv_tol_grad = TOL
    #mf.kernel()

    return mol, mf, frag_atms_list

def fragment(atoms, charge, basis="sto3g"):

    mol = pyscf.gto.Mole()
    mol.build(
        atom = atoms,
        charge = charge,
        basis = basis,
        verbose=0
    )

    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    mf.max_cycle = 100
    mf.conv_tol  = TOL
    mf.conv_tol_grad = TOL
    mf.kernel()

    return mf.mo_coeff, mf.mo_occ

def get_ao(mol, frag_atms_list):

    ao_slice_by_atom = mol.aoslice_by_atom()

    imp_atom = frag_atms_list[0]
    env_atom = frag_atms_list[1]
    imp_ao = []
    env_ao = []

    for i in imp_atom :
        ao_start, ao_end = ao_slice_by_atom[i,2:4]
        for j in range(ao_start,ao_end):
            imp_ao.append(j)

    for i in env_atom :
        ao_start, ao_end = ao_slice_by_atom[i,2:4]
        for j in range(ao_start,ao_end):
            env_ao.append(j)
    return imp_ao, env_ao

def get_dm(coeff, ovlp):
    CSC = numpy.einsum('ji,jk,kl->il', coeff, ovlp, coeff, optimize=True)
    CSC_inv = numpy.linalg.inv(CSC)
    dm = 2 * numpy.einsum('ij,jk,lk->il', coeff, CSC_inv, coeff, optimize=True)
    return dm

numpy.set_printoptions(precision=4)
mol, mf, frag_atms_list = build(basis="sto3g")
atom = mol.atom.split()
imp_atom = ''
env_atom = ''
for i in frag_atms_list[0] :
    imp_atom += atom[4*i] + ' '
    imp_atom += atom[4*i+1] + ' '
    imp_atom += atom[4*i+2] + ' '
    imp_atom += atom[4*i+3]
    imp_atom += '\n'
for i in frag_atms_list[1] :
    env_atom += atom[4*i] + ' '
    env_atom += atom[4*i+1] + ' '
    env_atom += atom[4*i+2] + ' '
    env_atom += atom[4*i+3]
    env_atom += '\n'

imp_ao, env_ao = get_ao(mol, frag_atms_list)
coeffa, mo_occa = fragment(atoms=imp_atom, charge=0, basis=mol.basis)
coeffe, mo_occe = fragment(atoms=env_atom, charge=0, basis=mol.basis)

nimp = len(imp_ao)
nenv = len(env_ao)

mo_occ = numpy.hstack((mo_occa, mo_occe))
occidxa = numpy.where(mo_occa==2)[0]
occidxe = numpy.where(mo_occe==2)[0]
occidx  = numpy.where(mo_occ==2)[0]

ovlp = mf.get_ovlp()
hcore = mf.get_hcore()
mo_coeff = numpy.zeros_like(ovlp)
imp_imp_ix = numpy.ix_(imp_ao, imp_ao)
env_env_ix = numpy.ix_(env_ao, env_ao)
mo_coeff[imp_imp_ix] = coeffa
mo_coeff[env_env_ix] = coeffe
dm0 = get_dm(mo_coeff[:,occidx], ovlp)
fock = mf.get_fock(dm=dm0)

ovlp_aa = ovlp[imp_ao,:][:,imp_ao]
ovlp_ae = ovlp[imp_ao,:][:,env_ao]
ovlp_ea = ovlp[env_ao,:][:,imp_ao]
ovlp_ee = ovlp[env_ao,:][:,env_ao]

maxcycle = 50
tol = 1e-6

for icycle in range(maxcycle):

    dma = get_dm(coeffa[:,occidxa], ovlp_aa)
    dme = get_dm(coeffe[:,occidxe], ovlp_ee)
    
    ovlpa = ovlp_aa - numpy.einsum('ij,jk,kl->', ovlp_ae, dme, ovlp_ea, optimize=True)
    ovlpe = ovlp_ee - numpy.einsum('ij,jk,kl->', ovlp_ea, dma, ovlp_ae, optimize=True)

    Ta  = numpy.hstack((numpy.eye(nimp), -numpy.einsum('ij,jk->ik', ovlp_ae, dme, optimize=True)))
    Tat = numpy.vstack((numpy.eye(nimp), -numpy.einsum('ij,jk->ik', dme, ovlp_ea, optimize=True)))
    focka = numpy.einsum('ij,jk,kl->il', Ta, fock, Tat, optimize=True)
    Te  = numpy.hstack((-numpy.einsum('ij,jk->ik', ovlp_ea, dma, optimize=True), numpy.eye(nenv)))
    Tet = numpy.vstack((-numpy.einsum('ij,jk->ik', dma, ovlp_ae, optimize=True), numpy.eye(nenv)))
    focke = numpy.einsum('ij,jk,kl->il', Te, fock, Tet, optimize=True)

    energya, coeffa = eigh(focka, ovlpa)
    energye, coeffe = eigh(focke, ovlpe)

    coeff = numpy.zeros_like(dm0)
    coeff[imp_imp_ix] = coeffa
    coeff[env_env_ix] = coeffe

    dm = get_dm(coeff[:,occidx], ovlp)
    fock = mf.get_fock(dm=dm)

    delta = numpy.linalg.norm(dm-dm0)
    Etot  = 0.5 * numpy.einsum('ij,ji->', dm, fock + hcore, optimize=True)
    if delta < tol :
        print(f'Cycle {icycle+1:2d} converage, delta dm = {delta:10.5g}, Etot = {Etot:16.10f} Hartree')
        break
    print(f'Cycle {icycle+1:2d}, delta dm = {delta:10.5g}, Etot = {Etot:16.10f} Hartree')
    dm0 = dm