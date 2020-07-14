import numpy as np 
from scipy.optimize import minimize, rosen, rosen_der

def find_degen(eigval, eps):
   group = np.zeros(eigval.shape, dtype=np.int)

   reference  = eigval[0]
   curr_group = 0 
   for i, e in enumerate(eigval):
      if(np.abs(e - reference) > eps):
         curr_group += 1 
         reference = e
      group[i] = curr_group
   return group

def check_linear_comb(spectrum, y, eps):
   n = spectrum.shape[1]

   f = lambda coeffs: np.linalg.norm((y - coeffs@spectrum.transpose()))
   res = minimize(f, np.ones(n)/np.sqrt(n), tol=eps)
   #print(res)
   return res.fun < 1e-6

def comp_spectrum(a_vec, a_val, b_vec, b_val, eps=1e-6):
   equal = True 

   if np.max(np.abs(a_val - b_val)) > eps:
      return False 

   groups = find_degen(a_val, eps) 

   for g in np.unique(groups):
      group_elem = np.argwhere(groups == g)[:,0]
      for elem in group_elem:
         if(not check_linear_comb(a_vec[:,group_elem], b_vec[:,elem], eps)):
            print("first: ", group_elem, elem)
            return False
            
         if(not check_linear_comb(b_vec[:,group_elem], a_vec[:,elem], eps)):
            print("second: ", group_elem, elem)
            return False
         print(f"{elem} ok")

   return True
   
def list_non_deg(eigval, eps=1e-6):
   groups = find_degen(eigval, eps) 
   unique, counts = np.unique(groups, return_counts=True)

   single_groups = np.argwhere(counts == 1)[:,0]
   single_idx = []
   for i in single_groups:
      single_idx.append(np.argwhere(groups == unique[i])[0,0])
   
   return np.array(single_idx)

eigvec1 = np.load("/home/matthias/calculation/debug/mpi=1/zmat_kqpt=1_bandoi=1.npy")
eigvec2 = np.load("/home/matthias/calculation/debug/mpi=2/zmat_kqpt=1_bandoi=1.npy")

eigval1 = np.load("/home/matthias/calculation/debug/mpi=1/eigval_nk=1.npy")
eigval2 = np.load("/home/matthias/calculation/debug/mpi=2/eigval_nk=1.npy")

psi1 = np.load("/home/matthias/calculation/debug/mpi=1/psi_kqpt_nk=1_iq=1_bandoi=1.npy")
psi2a = np.load("/home/matthias/calculation/debug/mpi=2/psi_kqpt_nk=1_iq=1_bandoi=1.npy")
psi2b = np.load("/home/matthias/calculation/debug/mpi=2/psi_kqpt_nk=1_iq=1_bandoi=5.npy")
psi2 = np.hstack([psi2a, psi2b])

cprod1 = np.load("/home/matthias/calculation/debug/mpi=1/cprod_is_nk=1_iq=1_bandoi=1.npy")
iob1   = np.load("/home/matthias/calculation/debug/mpi=1/iob_is_nk=1_iq=1_bandoi=1.npy")
iband1 = np.load("/home/matthias/calculation/debug/mpi=1/iband_is_nk=1_iq=1_bandoi=1.npy")

cprod2a = np.load("/home/matthias/calculation/debug/mpi=2/cprod_is_nk=1_iq=1_bandoi=1.npy")
iob2a  = np.load("/home/matthias/calculation/debug/mpi=2/iob_is_nk=1_iq=1_bandoi=1.npy")
iband2a = np.load("/home/matthias/calculation/debug/mpi=2/iband_is_nk=1_iq=1_bandoi=1.npy")

cprod2b = np.load("/home/matthias/calculation/debug/mpi=2/cprod_is_nk=1_iq=1_bandoi=5.npy")
iob2b  = np.load("/home/matthias/calculation/debug/mpi=2/iob_is_nk=1_iq=1_bandoi=5.npy")
iband2b = np.load("/home/matthias/calculation/debug/mpi=2/iband_is_nk=1_iq=1_bandoi=5.npy")

cprod2 = np.hstack([cprod2a, cprod2b])
iob2   = np.hstack([iob2a, iob2b])
iband2 = np.hstack([iband2a, iband2b])

non_deg = list_non_deg(eigval1)

for i in range(312):
   print(iob1[i], iband1[i], "###", iob2[i], iband2[i])