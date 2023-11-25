import torch
import torch.nn as nn
import numpy as np
import vqe_helper_functions as hf
import pennylane as qml
from scipy.sparse import linalg

# dtype = torch.complex128
dtype = torch.complex64
device = torch.device("cpu")
torch.manual_seed(0)

M = 3  # max number of fragment qubits per fragment
a = 2  # number of auxiliary qubits per fragment)
T = 1  # number of repeated optimizations (feel free to change the number; I set it to 2 just to speed things up)
spin_n = 6  # N (size of system)

# transverse field (set to classical case for now):
h = 0.0
hi = torch.mul(torch.ones(spin_n, device=device), h)

# Jij "graph"
J = 1.0;
connectivity = 'All to All'
if connectivity == 'All to All':
    nonzero = torch.triu(torch.ones((spin_n, spin_n), device=device)) - torch.diag(torch.ones(spin_n, device=device), diagonal=0)
elif connectivity == 'Nearest Neighbor':
    nonzero = torch.diag(torch.ones(spin_n - 1, device=device), diagonal=1)

# you can play around with this if you want different Jij (this is just plain old Gaussian all-to-all)
vals = torch.normal(0, J, size=(spin_n, spin_n), device=device)
Jij_U = vals * nonzero
Jij = Jij_U + torch.transpose(Jij_U, 0, 1)  # should always be symmetric

spin_list = torch.multiply(torch.ones(spin_n, device=device), 0.5)
H_dim = (1 << spin_n)
spin_ind = torch.arange(spin_n, device=device)

# hyper parameters of PQC:
layers_shallow = 4  # number of linear entangling layers used in pre-training
layers_complex = spin_n  # number of all-to-all entangling layers in full circuit
#rots = ['Rx', 'Ry']
rots = torch.tensor([1, 2], device=device)  # 1: Rx, 2: Ry, 3: Rz
num_rot = len(rots)
ent_gate = 3  # 1: CNOT, 2: CY, 3: CZ
max_iter = 5000

# dev = qml.device('lightning.gpu', wires=spin_n)
dev = qml.device('default.qubit', wires=spin_n)

@qml.qnode(dev, interface='torch')
def linear_loss(sub_H, params_frag, rots, layers, frag_and_aux, ent_gate=1):
    hf.get_linear_circuit(params_frag, rots, layers, frag_and_aux, ent_gate=ent_gate)
    return qml.expval(sub_H)

@qml.qnode(dev, interface='torch')
def linear_loss_MF(coeffs, obs, Jij, params_frag, rots, layers, sz_vec, aux_ind, target_ind,
                   target_neighbors, frag_and_aux, model='Ising',ent_gate=1):
    hf.get_linear_circuit(params_frag, rots, layers, frag_and_aux, ent_gate=ent_gate)
    if model == 'Ising':
        mf_coeffs, mf_obs = hf.get_ising_MF_correction_terms_qml(Jij, sz_vec, aux_ind, target_ind, target_neighbors)
    sub_H_MF = qml.Hamiltonian(coeffs + mf_coeffs, obs + mf_obs)
    return qml.expval(sub_H_MF)

@qml.qnode(dev, interface='torch')
def sz_update(params_frag, rots, layers, frag_and_aux, ent_gate=1):
    hf.get_linear_circuit(params_frag, rots, layers, frag_and_aux, ent_gate=ent_gate)
    return [qml.expval((1 / 2) * qml.PauliZ(x.item())) for x in frag_and_aux]

@qml.qnode(dev, interface='torch')
def complex_loss(H, params_shallow, params_complex, rots, layers_shallow, layers_complex, spin_ind, spin_n, ent_gate=1):
    hf.get_linear_circuit(params_shallow, rots, layers_shallow, spin_ind, ent_gate=ent_gate)
    combinations = hf.get_interactions(spin_ind, spin_n)
    param_index = 0
    for _ in np.arange(layers_complex):
        for combo in combinations:
            if ent_gate == 1:
                qml.CNOT(wires=[combo[0].item(), combo[1].item()])
            elif ent_gate == 2:
                qml.CY(wires=[combo[0].item(), combo[1].item()])
            elif ent_gate == 3:
                qml.CZ(wires=[combo[0].item(), combo[1].item()])

        for rot in rots:
            for qubit in spin_ind:
                if rot == 3:
                    qml.RZ(params_complex[param_index], wires=qubit.item())
                elif rot == 2:
                    qml.RY(params_complex[param_index], wires=qubit.item())
                elif rot == 1:
                    qml.RX(params_complex[param_index], wires=qubit.item())
                param_index += 1

    return qml.expval(H)

# total number of parameters for "shallow" (linear entangling) section of circuit, and "complex" (all-to-all entangling) section of circuit
N_param_tot_shallow = num_rot * spin_n * (layers_shallow + 1)
N_param_tot_complex = num_rot * spin_n * (layers_complex)

model = "Ising"  # always
eps = 0.0001  # for complex circuit initialization after pre-training
psi0 = 0  # initialize to all |0>
H, H_coeffs, H_obs = hf.model_ham(spin_ind, spin_n, Jij, hi, model=model)

# Here, we calculate the exact g.s. energy using a scipy sparse matrix. This is computationally expensive, and won't be
# possible for very large systems (but the full circuit optimization will also be difficult for very large systems)
Ham = hf.create_sparse_ising_ham(Jij, hi, spin_n, spin_ind)
E0 = torch.real(torch.tensor(
    linalg.eigs(Ham, k=1, M=None, sigma=None, which='SR', v0=None, ncv=None, maxiter=None, tol=0,
                return_eigenvectors=False, Minv=None, OPinv=None, OPpart=None), device=device))

print('ED energy: ' + str(E0.item()))

# Ok, first, we pre-train. A total of T partitions are generated and pre-trained; these are independent and could be
# parallelized (just need to save optimal parameters from each pre-training).
# (also, fewer resources are required for this than for the full circuit optimization, which is another reason to
# pre-train in a separate simulation)
list_of_params = torch.zeros((T, N_param_tot_shallow), device=device)
list_of_initial_E = torch.zeros(T, device=device)
for trial in range(T):
    count = 0
    overlap = 1.0
    sz_vec = torch.zeros(spin_n, dtype=torch.float64, device=device)
    params_last = torch.ones(N_param_tot_shallow, device=device)
    thresh = 1e-3
    best_est_loss = 100
    params_current = torch.rand(N_param_tot_shallow, device=device)

    # generate a random partition of the circuit:
    frag_list, aux_list, num_frag, frag_range = hf.gen_partition(spin_n, M, a)
    frag_N_list, frag_sys_N_list, aux_target_list, frag_and_aux_list, aux_target_neighbors_list = hf.get_frag_info_circ(
        spin_n, spin_ind, frag_list, aux_list, num_frag)
    sub_Ham_list, coeff_list, obs_list = hf.get_sub_H_info(frag_and_aux_list, frag_N_list, frag_range, Jij, hi, model=model)
    N_params_frag_list = torch.tensor([num_rot * frag_N_list[x] * (layers_shallow + 1) for x in frag_range], device=device)
    loss_vec_MFlist = torch.zeros((num_frag, max_iter), device=device) + float('nan')

    # splitting up parameters of full circuit into sub-circuit parameters
    params_frag_list = torch.zeros((num_frag, torch.max(N_params_frag_list)), device=device)
    params_frag_ind = torch.zeros(num_frag, dtype=torch.int64, device=device)
    for layer in torch.arange((layers_shallow + 1) * num_rot, device=device):
        n_ind = 0
        for ii in frag_range:
            start_ind = layer * spin_n + n_ind
            if ii > 0:
                start_ind -= aux_list[ii]
            params_frag_list[ii, params_frag_ind[ii]:params_frag_ind[ii] + frag_N_list[ii]] = params_current[start_ind:start_ind + frag_N_list[ii]]
            n_ind += frag_sys_N_list[ii]
            params_frag_ind[ii] += frag_N_list[ii]

    while (overlap > thresh) and (count < max_iter):
        spin_index = 0
        for i in frag_range:
            # Just fetching some info about this particular fragmented circuit:
            N_param_frag = N_params_frag_list[i]
            frag_and_aux = frag_and_aux_list[i]
            frag = frag_list[i]
            coeffs = coeff_list[i]
            obs = obs_list[i]
            frag_N = frag_N_list[i]
            frag_sys_n = frag_sys_N_list[i]
            aux_ind = torch.arange(frag_N)
            target_ind = frag_and_aux  # we mean-field correct auxiliaries and fragment system qubits alike
            target_neighbors = aux_target_neighbors_list[i]

            # Fetch most updated values from params_current:
            frag_ind = 0
            for layer in np.arange((layers_shallow + 1) * num_rot):
                start_ind = layer * spin_n + spin_index
                if i > 0:
                    start_ind -= aux_list[i]
                params_frag_list[i, frag_ind:frag_ind + frag_N] = params_current[start_ind:start_ind + frag_N]
                frag_ind += frag_N

            # Formally initializing parameters
            params_frag = nn.Parameter(params_frag_list[i, :N_param_frag], requires_grad=True)
            optimizer = torch.optim.Adam([params_frag], lr=1e-3)

            # Calculate loss:
            loss = linear_loss_MF(coeffs, obs, Jij, params_frag, rots, layers_shallow, sz_vec, aux_ind, target_ind,
                                  target_neighbors, frag_and_aux, model=model, ent_gate=ent_gate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_vec_MFlist[i, count] = loss.detach()

            # Update mean-fields:
            # (being careful about auxiliary qubits -- I only want to update the "core" registers,
            # not auxiliary target indices)
            sz_frag = sz_update(params_frag, rots, layers_shallow, frag_and_aux, ent_gate=ent_gate)
            start_ind = 0
            if i > 0:
                start_ind += aux_list[i]
            sz_vec[frag] = torch.tensor(sz_frag)[start_ind:start_ind + frag_sys_n]

            # Store updated params into params_current:
            frag_ind = 0
            for layer in np.arange((layers_shallow + 1) * num_rot):
                start_ind = layer * spin_n + spin_index
                if i > 0:
                    start_ind -= aux_list[i]
                params_current[start_ind:start_ind + frag_N] = params_frag.detach()[frag_ind:frag_ind + frag_N]
                frag_ind += frag_N

            # Shifting to next fragment:
            spin_index += frag_sys_n

        if np.mod(count, 100) == 0:
            # check if parameters have reached steady state:
            overlap = torch.sum(torch.abs(params_last - params_current)) / (N_param_tot_shallow)
            params_last = params_current.clone()
            print(loss_vec_MFlist[:, count])
        count += 1

    list_of_params[trial, :] = params_current.detach().clone()
    #params_tot_shallow = nn.Parameter(torch.tensor(params_current.detach(), device=device), requires_grad=True)
    # technically here we use the full circuit, but if the parameters are stored, this could be done as part of the full circuit simulation
    loss = linear_loss(H, params_current.detach(), rots, layers_shallow, spin_ind, ent_gate=ent_gate)
    print("Loss after pre-training trial " + str(trial) + ": " + str(loss.item()))
    list_of_initial_E[trial] = loss.detach()

params_tot_MFinit = list_of_params[torch.argmin(list_of_initial_E)]

# Ok, now we move to the section that requires the full circuit
########################################################################################
############# Full Optimization w/ Best Pre-Training: ##################################
########################################################################################
# Use optimized parameters from best pre-training attempt to initialize full circuit:
params_tot_shallow = nn.Parameter(params_tot_MFinit.detach(), requires_grad=True)
loss = linear_loss(H, params_tot_shallow, rots, layers_shallow, spin_ind, ent_gate=ent_gate)
percent_error = (loss.detach() - E0) / torch.abs(E0)
print("Best initial error after pre-training: " + str(percent_error.item()))
# Use near-identity initialization for all-to-all entangling section of circuit (nearly zeros work for now, but won't produce identity for for all circuit architectures)
params_tot_complex = nn.Parameter(torch.rand(N_param_tot_complex, device=device) * eps - eps / 2, requires_grad=True)
optimizer = torch.optim.Adam([params_tot_shallow, params_tot_complex], lr=1e-3)
loss_vec_tot = torch.zeros(max_iter, device=device) + float('nan')
count = 0
overlap = 1.0
thresh = 1e-3
params_last = torch.ones(N_param_tot_complex + N_param_tot_shallow, device=device)
while (overlap > thresh) and (count < max_iter):
    loss = complex_loss(H, params_tot_shallow, params_tot_complex, rots, layers_shallow, layers_complex, spin_ind,
                        spin_n, ent_gate=ent_gate)
    loss_vec_tot[count] = loss.detach()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if np.mod(count, 100) == 0:
        # check if parameters have reached steady state:
        params_current = torch.cat((params_tot_shallow.clone().detach(), params_tot_complex.clone().detach()))
        overlap = torch.sum(torch.abs(params_last - params_current)) / (N_param_tot_shallow + N_param_tot_complex)
        params_last = params_current.clone()
    count += 1
loss_vec_tot = loss_vec_tot[~loss_vec_tot.isnan()]
percent_error_tot = (loss_vec_tot - E0) / torch.abs(E0)
print("Final error after pre-training: " + str(percent_error_tot[-1].item()))

########################################################################################
############# Random Case ("Vanilla VQE"): #############################################
########################################################################################
# Random initialization:
params_tot_shallow_rand = nn.Parameter(torch.rand(N_param_tot_shallow, device=device), requires_grad=True)
params_tot_complex_rand = nn.Parameter(torch.rand(N_param_tot_complex, device=device), requires_grad=True)
optimizer = torch.optim.Adam([params_tot_shallow_rand, params_tot_complex_rand], lr=1e-3)
loss_vec_tot_rand = torch.zeros(max_iter, device=device) + float('nan')
count = 0
overlap = 1.0
thresh = 1e-3
params_last = torch.ones(N_param_tot_complex + N_param_tot_shallow, device=device)
while (overlap > thresh) and (count < max_iter):
    loss = complex_loss(H, params_tot_shallow_rand, params_tot_complex_rand, rots, layers_shallow, layers_complex,
                        spin_ind, spin_n, ent_gate=ent_gate)
    loss_vec_tot_rand[count] = loss.detach()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if np.mod(count, 100) == 0:
        # check if parameters have reached steady state:
        params_current = torch.cat((params_tot_shallow_rand.clone().detach(), params_tot_complex_rand.clone().detach()))
        overlap = torch.sum(torch.abs(params_last - params_current)) / (N_param_tot_shallow + N_param_tot_complex)
        params_last = params_current.clone()
    count += 1

loss_vec_tot_rand = loss_vec_tot_rand[~loss_vec_tot_rand.isnan()]
percent_error_tot_rand = (loss_vec_tot_rand - E0) / torch.abs(E0)
print("Final error for Vanilla VQE: " + str(percent_error_tot_rand[-1].item()))

