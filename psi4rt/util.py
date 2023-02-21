import numpy as np
import psi4
import sys
import torch

from pkg_resources import parse_version

import rtutil

##################################################################

def set_input(fgeom):
  geomobj = str()
  with open(fgeom,"r") as f:
   next(f)
   next(f)
   for line in f:
    geomobj +=str(line)
  geomobj += "symmetry c1" +"\n" +"no_reorient" +"\n" +"no_com"
  print(geomobj)
  mol = psi4.geometry(geomobj)
  f.close()
  return geomobj, mol

##################################################################

def exp_opmat(mat,dt):

    #first find eigenvectors and eigenvalues of F mat
    try:
       w,v=torch.linalg.eigh(mat)
    except torch.linalg.LinAlgError:
        print("Error in torch.linalg.eigh of inputted matrix")
        return None

    diag=torch.exp(-1.j*w*dt)

    dmat=torch.diagflat(diag)

    # for a general matrix Diag = M^(-1) A M
    # M is v
    #try:
    #   v_i=torch.linalg.inv(v)
    #except torch.linalg.LinAlgError:
    #   return None

    # transform back
    #tmp = torch.matmul(dmat,v_i)
    tmp = torch.matmul(dmat,torch.conj(v.T))

    #in an orthonrmal basis v_inv = v.H

    mat_exp = torch.matmul(v,tmp)

    return mat_exp

##################################################################

def get_Fock(D, Hcore, I, f_type, basisset):
    # Build J,K matrices
    J = torch.einsum('pqrs,rs->pq', I.type(torch.complex128), D.type(torch.complex128))
    if (f_type=='hf'):
        K = torch.einsum('prqs,rs->pq', I, D)
        F = Hcore + J*np.float_(2.0) - K
        Exc=0.0
        J_ene = 0.0
    else:
        #D must be a psi4.core.Matrix object not a numpy.narray
        restricted = True
        if parse_version(psi4.__version__) >= parse_version('1.3a1'):
                build_superfunctional = psi4.driver.dft.build_superfunctional
        else:
                build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
        sup = build_superfunctional(f_type, restricted)[0]
        sup.set_deriv(2)
        sup.allocate()
        vname = "RV"
        if not restricted:
            vname = "UV"
        potential=psi4.core.VBase.build(basisset,sup,vname)
        Dm=psi4.core.Matrix.from_array(D.real)
        potential.initialize()
        potential.set_D([Dm])
        nbf=D.shape[0]
        V=psi4.core.Matrix(nbf,nbf)
        potential.compute_V([V])
        potential.finalize()
        F = Hcore + J*np.float_(2.0) + torch.from_numpy(V.to_array())
        Exc= potential.quadrature_values()["FUNCTIONAL"]
        if sup.is_x_hybrid():
          alpha = sup.x_alpha()
          K = torch.einsum('prqs,rs->pq', I.type(torch.complex128), D.type(torch.complex128))
          F += -alpha*K
          Exc += -alpha*torch.trace(torch.matmul(D,K))
        J_ene=2.00*torch.trace(torch.matmul(D,J))
    return J_ene,Exc,F

##################################################################

def set_params(filename="input.inp"):

    my_dict = {}
    with open(filename) as fileobj:
      for line in fileobj:
        key, value = line.split(":")
        my_dict[key.strip()] = value.strip()
    fileobj.close()
    imp_params = {}
    imp_params['Fmax'] = float(my_dict['F_max'])
    imp_params['w'] = float(my_dict['freq_carrier'])
    imp_params['s'] = float(my_dict['sigma'])
    imp_params['t0'] = float(my_dict['t0']) 
    imp_params['imp_type'] = my_dict['imp_type']
    
    calc_params ={}    
    calc_params['time_int']=float(my_dict['time_int'])
    calc_params['delta_t']=float(my_dict['delta_t'])
    calc_params['func_type'] =my_dict['func_type'] 
    calc_params['method']=my_dict['method_type']
    return imp_params,calc_params

##################################################################

def mo_fock_mid_forwd_eval(D_ti,fock_mid_ti_backwd,i,delta_t,H,I,dipole,\
                               C,C_inv,S,nbf,imp_opts,f_type,fout,basisset,\
                               extpotin = np.array([None])):

    extpot = extpotin
    
    if hasattr(extpotin, "__len__"):
      if extpotin.all() == None:
        extpot = 0
    else:
      if extpotin == None:
        extpot = 0

    t_arg=np.float_(i)*np.float_(delta_t)
    
    func = rtutil.funcswitcher.get(imp_opts['imp_type'], lambda: rtutil.kick)
    
    pulse = func(imp_opts['Fmax'], imp_opts['w'], t_arg,\
                        imp_opts['t0'], imp_opts['s'])

    #D_ti is in AO basis
    #transform in the MO ref basis
    Dp_ti= torch.matmul(C_inv,torch.matmul(D_ti,torch.conj(C_inv.T)))
    
    k=1
    
    J_i,Exc_i,fock_mtx=get_Fock(D_ti,H,I,f_type,basisset)
    #add -pulse*dipole
    fock_ti_ao = fock_mtx - (dipole*pulse)

    #if i==0:
    #    print('F(0) equal to F_ref: %s' % torch.allclose(fock_ti_ao,fock_mid_ti_backwd))
    
    #initialize dens_test !useless
    dens_test=torch.zeros(Dp_ti.shape)

    # set guess for initial fock matrix
    fock_guess = None
    if torch.is_tensor(fock_mid_ti_backwd) is False:  
        fock_mid_ti_backwd=torch.from_numpy(fock_mid_ti_backwd.to_array())
    else:
        None
    #print(type(fock_ti_ao),type(extpot),type(fock_mid_ti_backwd))
    fock_guess = 2.00*( fock_ti_ao + extpot ) - fock_mid_ti_backwd
    #if i==0:
    #   print('Fock_guess for i =0 is Fock_0: %s' % torch.allclose(fock_guess,fock_ti_ao))
    #transform fock_guess in MO basis
    while True:
        fockp_guess=torch.matmul(torch.conj(C.T),torch.matmul(fock_guess,C))
        u=exp_opmat(fockp_guess,delta_t)
        #u=scipy.linalg.expm(-1.j*fockp_guess*delta_t) ! alternative routine
        test=torch.matmul(u,torch.conj(u.T))
        #print('U is unitary? %s' % (torch.allclose(test,torch.eye(u.shape[0]))))
        if (not torch.allclose(test,torch.eye(u.shape[0]).type(torch.complex128))):
            Id=torch.eye(u.shape[0])
            diff_u=test-Id
            norm_diff=torch.linalg.norm(diff_u,'fro')
            fout.write('fock_mid:U deviates from unitarity, |UU^-1 -I| %.8f' % norm_diff)
        #evolve Dp_ti using u and obtain Dp_ti_dt (i.e Dp(ti+dt)). u i s built from the guess fock
        #density in the orthonormal basis
        tmpd=torch.matmul(Dp_ti,torch.conj(u.T))
        Dp_ti_dt=torch.matmul(u,tmpd)
        #backtrasform Dp_ti_dt
        D_ti_dt=torch.matmul(C,torch.matmul(Dp_ti_dt,torch.conj(C.T)))
        #build the correspondig Fock : fock_ti+dt
        
        dum1,dum2,fock_mtx=get_Fock(D_ti_dt,H,I,f_type,basisset)
        #update t_arg+=delta_t
        pulse_dt = func(imp_opts['Fmax'], imp_opts['w'], t_arg+delta_t,\
                        imp_opts['t0'], imp_opts['s'])
        fock_ti_dt_ao=fock_mtx -(dipole*pulse_dt)
        fock_inter= 0.5*fock_ti_ao + 0.5*fock_ti_dt_ao + extpot
        #update fock_guess
        fock_guess=torch.clone(fock_inter)
        if k >1:
        #test on the norm: compare the density at current step and previous step
        #calc frobenius of the difference D_ti_dt_mo_new-D_ti_dt_mo
            diff=D_ti_dt-dens_test
            norm_f=torch.linalg.norm(diff,'fro')
            if norm_f<(1e-6):
                tr_dt=torch.trace(torch.matmul(S.type(torch.complex128),D_ti_dt))
                fout.write('converged after %i interpolations\n' % (k-1))
                fout.write('i is: %d\n' % i)
                fout.write('norm is: %.12f\n' % norm_f)
                fout.write('Trace(D)(t+dt) : %.8f\n' % tr_dt.real)
                break
        dens_test=torch.clone(D_ti_dt)
        k+=1
        if k > 20:
         raise Exception("Numember of iterations exceeded (k>20)")
    return J_i,Exc_i,pulse,fock_ti_ao,fock_inter

##################################################################
# analysis based on MO-weighted dipole

def dipoleanalysis(dipole,dmat,nocc,occlist,virtlist,debug=False,HL=False):
    #virtlist can also contain occupied orbitals !check
    #just HOMO-LUMO vertical transition
    tot = len(occlist)*len(virtlist)
    if HL:
      i = nocc
      a = nocc+1
      res = dipole[i-1,a-1]*dmat[a-1,i-1] + dipole[a-1,i-1]*dmat[i-1,a-1]
    else:
      res = torch.zeros(tot,dtype=torch.complex128)
      count = 0
      for i in occlist:
        for j in virtlist:
           res[count] = dipole[i-1,j-1]*dmat[j-1,i-1] + dipole[j-1,i-1]*dmat[i-1,j-1]
           count +=1

    return res


#######################################################################
def dipole_selection(dipole,ID,nocc,occlist,virtlist,odbg=sys.stderr,debug=False):
    
    if debug:
       odbg.write("Selected occ. Mo: %s \n"% str(occlist))
       odbg.write("Selected virt. Mo: %s \n"% str(virtlist))
    offdiag = torch.zeros_like(dipole)
    #diag = numpy.diagonal(tmp)
    #diagonal = numpy.diagflat(diag)
    nvirt = dipole.shape[0]-nocc
    odbg.write("n. virtual orbitals : %i\n" % nvirt)
    if (ID == 99):
      for b in range(nvirt):
        for j in occlist:
          offdiag[nocc+b,j-1] = dipole[nocc+b,j-1]
    else:
      for b in virtlist:
        for j in  occlist:
          offdiag[b-1,j-1] = dipole[b-1,j-1]
    offdiag=(offdiag+torch.conjugate(offdiag.T))
    #offdiag+=diagonal
    res = offdiag

    return res
