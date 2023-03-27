#import sys
#
#sys.path.append("/home/matteod/build/xcfun/build/lib/python")
#sys.path.append("/home/matteod/psi4conda/lib/python3.7")
#sys.path.append("/home/matteod/pybertha/psi4rt")
#sys.path.append("/home/matteod/pybertha/src")
#sys.path.append("/home/matteod/build/pyadf/src")

import os
import psi4
import scipy.linalg
import numpy as np
import pyadf
from pyadf.Plot.GridFunctions import GridFunctionFactory
import torch
print(torch.cuda.is_available())


def fcorr(emb_pot,dens_act,dens_env,alpha=0.1):
    scaled_env = dens_env.__mul__(alpha)
    temp = dens_act.__div__(scaled_env)
    arg = temp.__pow__(2.0)
    arg = arg.__mul__(-1.0)
    factor = arg.apply_function(np.exp)
    tcorr = emb_pot.__mul__(factor)
    corrected =  emb_pot.__sub__(tcorr)
    return corrected


#read grid (xyzw) from file and a property if potjob=True
def read_gridfunction(fname, potjob=False):
  if potjob:
    col = 5
  else:
   col = 4
  j = 0
  with open(fname,"r") as f:
   nlines = int(next(f))
   list = np.zeros((nlines,col),dtype=np.float_)
   for line in f:
    raw = line.split()
    list[j,:]=raw
    j += 1

  if potjob:
    prop = np.array(list[0:,4])
    xs = 0
    ys = 0
    zs = 0
    ws = 0
  else:
    # storing xyzw and prop
    xs = psi4.core.Vector.from_array(list[0:, 0])
    ys = psi4.core.Vector.from_array(list[0:, 1])
    zs = psi4.core.Vector.from_array(list[0:, 2])
    ws = psi4.core.Vector.from_array(list[0:, 3])
    prop = 0
  f.close()
  return xs, ys, zs, ws, prop

#plot the grid
def grid_plot(xs,ys,zs,ws): 
#plot the grid 
 R = np.sqrt(np.array(xs)**2 + np.array(ys)**2 + np.array(zs)**2)
 fig , ax =plt.subplots()
 ax.scatter(np.array(xs),np.array(ys),c=np.array(ws))
 fig.savefig('./grid.png')
 plt.close(fig)

def set_input(fgeom,basis_set,fgeomB,ghostB):
  geomobj = str()
  with open(fgeom,"r") as f:
   next(f)
   next(f)
   for line in f:
    geomobj +=str(line)
  f.close()

  if ghostB:
    ghost_str= str()
    with open(fgeomB,"r") as f:
     next(f)
     next(f)
     for line in f:
      ghost_str +=str(line)
    f.close()
    tmp=ghost_str.split('\n')
    tmp.pop()
    for m in tmp:
        geomobj+="@"+m.strip()+'\n'

  geomobj += "symmetry c1" +"\n" +"no_reorient" +"\n" +"no_com"
  #print(geomobj)
  mol =psi4.geometry(geomobj)
  psi4.set_options({'BASIS': basis_set,
                    'puream' : 'True',
                    'dft_radial_scheme' : 'becke',
                    #'dft_radial_points': ,
                    'dft_spherical_points' : 434,  #'dft_nuclear_scheme': 'treutler' | default
                    'scf_type' : 'direct',
                    #'DF_BASIS_SCF': 'cc-pvqz-jkfit',
                    'DF_SCF_GUESS': 'False',
                     'CUBIC_GRID_OVERAGE' : [7.0,7.0,7.0],
                     'CUBIC_GRID_SPACING' : [0.1,0.1,0.1],
                     'cubeprop_tasks': ['density'],
                     'CUBEPROP_ISOCONTOUR_THRESHOLD' : 1.0,
                    'd_convergence' : 1.0e-8,
                    'e_convergence' : 1.0e-8}) #more options if needed
  return geomobj, mol

def get_elpot(xs,ys,zs,Dnew=None,wfn=None,update=False):
  if (update):
    Dreal = Dnew.real
    wfn.oeprop.set_Da_ao(psi4.core.Matrix.from_array(Dreal))
    wfn.oeprop.compute()
    Vvals = wfn.oeprop.Vvals()
  else: 
    with open('grid.dat', 'w') as fout:
        for i in range(xs.np.shape[0]):
            fout.write("{: 21.18e} {: 21.18e} {: 21.18e}\n".format(xs.np[i], ys.np[i], zs.np[i]))
    fout.close() 
    ene,wfn = psi4.prop('blyp', properties=['GRID_ESP'], return_wfn=True) 
    #ene,wfn = psi4.energy('blyp',return_wfn=True)
 
    Vvals = wfn.oeprop.Vvals()
  return Vvals, wfn

# hinweis : puream basis set has to be used
def phi_builder(xs,ys,zs,ws,basisobj):
  
  delta = 1.0e-2

  basis_extents = psi4.core.BasisExtents(basisobj,delta)

  blockopoints = psi4.core.BlockOPoints(xs, ys, zs, ws,basis_extents)
  npoints = blockopoints.npoints()
  print("n points: %i" % npoints)
  #needed?
  lpos = np.array(blockopoints.functions_local_to_global())

  #print("Local basis function mapping")
  #print(lpos)
  #print("lpos shape (%i,)" % (lpos.shape[0]))

  #print some info
  blockopoints.print_out('b_info.txt')

  nbas = basisobj.nbf() #number of basis functions

  funcs = psi4.core.BasisFunctions(basisobj,npoints,nbas)

  funcs.compute_functions(blockopoints)

  phi = np.array(funcs.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
  return phi, lpos,nbas

def embpot2mat(phi,nbas,gfunc,ws,lpos):
    #compute  pot matrix
    res_ = np.zeros((nbas,nbas),dtype=np.complex128) # check
    phi=torch.from_numpy(phi)
    gfunc=torch.from_numpy(gfunc)
    ws=torch.from_numpy(ws.to_array())
    tmp = torch.einsum('pb,p,p,pa->ab', phi, gfunc, ws, phi)
    #to check
    # Add the temporary back to the larger array by indexing, ensure it is symmetric
    tmp_=(tmp + torch.conj(tmp.T))
    res_[(lpos[:, None], lpos)] += 0.5 * tmp_.cpu().detach().numpy()
    res=torch.from_numpy(res_)
    #check Vtmp and V
    er = torch.allclose(res,tmp.type(torch.complex128),atol=1.0e-12)
    if  (not er):
      print("Check Vtmp and V")
    #print("N. basis funcs: %i\n" % nbas)
    #check if V has imag part
    if False:
       print("V is real: %s\n" % (torch.allclose(res.imag,torch.zeros((nbas,nbas)).type(torch.float),atol=1.0e-12)))
       np.savetxt("vemb.txt", res.real)
    return res.cpu().detach().numpy().real

def denstogrid(phi,D,S,ndocc):
#    S=S.to_array()
#    temp = torch.matmul(torch.from_numpy(S),torch.matmul(torch.from_numpy(D).real,torch.from_numpy(S)))
    temp = np.matmul(S,np.matmul(D.real,S))
    try:
       eigvals,eigvecs=scipy.linalg.eigh(temp,S,eigvals_only=False)
#       eigvals,eigvecs=torch.linalg.eigh(temp,torch.from_numpy(S),eigvals_only=False)
#        eigvals, eigvecs = torch.lobpcg(A=temp, B=torch.from_numpy(S), k=int(temp.size()[0]/3)
#                                         ,method='ortho')
    except torch.linalg.LinAlgError:
        print("Error in torch.linalg.eigh of inputted matrix")
        return None
#    eigvecs_=eigvecs.cpu().detach().numpy()
    Rocc = eigvecs[:,-ndocc:]
    Rocc = torch.from_numpy(Rocc)
    MO = torch.matmul(torch.from_numpy(phi),Rocc)
    MOs = torch.square(MO)
    rho = torch.einsum('pm->p',MOs)
    return rho.cpu().detach().numpy()
