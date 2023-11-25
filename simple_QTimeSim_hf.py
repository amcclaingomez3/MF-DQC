import torch
import numpy as np
import pennylane as qml
import TimeSim_helper_functions as hf

#dtype = torch.complex128
dtype = torch.complex64
device = torch.device("cpu")

# Fragments

spin_n = 9  # this is big N
frag_partition = torch.tensor([3, 3, 3], device=device)  # this is how the N qubits are distributed between fragments
spin_range = torch.arange(spin_n, device=device)
num_frag = frag_partition.size(dim=0)
frag_range = torch.arange(num_frag, device=device)
frag_list = torch.nested.nested_tensor([torch.arange(frag_partition[x], device=device) + torch.sum(frag_partition[:x])
                                        for x in frag_range], device=device)

# The way things are set up, for interior fragments, the number of auxiliaries gets multiplied by 2
# (so this is really N_a = 2 for all fragments)
aux_list = torch.tensor([2, 1, 2], device=device)
# number of "system" qubits in the fragmentsspin_ind = torch.arange(spin_n, device=device)
frag_sys_N_list = torch.tensor([frag_list[x].size(dim=0) for x in torch.arange(num_frag)], device=device)
# total number of qubits in the fragments (system and auxiliary)
frag_N_list = torch.tensor([frag_sys_N_list[x] + aux_list[x] if x == 0 or x == num_frag - 1
                            else frag_sys_N_list[x] + int(2 * aux_list[x]) for x in torch.arange(num_frag)],
                           device=device)
spin_list = torch.multiply(torch.ones(spin_n, device=device), 0.5) # always qubits (at this point)

# transverse field:
h = 1.0
hi = torch.mul(torch.ones(spin_n, device=device), h)

# Jij "graph"
J = 1.0;
connectivity = 'All to All'
if connectivity == 'All to All':
    nonzero = torch.triu(torch.ones((spin_n, spin_n), device=device)) - torch.diag(torch.ones(spin_n, device=device), diagonal=0)
elif connectivity == 'Nearest Neighbor':
    nonzero = torch.diag(torch.ones(spin_n - 1, device=device), diagonal=1)

# you can play around with this if you want different Jij (this is just plain old Gaussian all-to-all)
vals = torch.normal(0, J, size=(spin_n, spin_n), device=device)   # (this is random Gaussian)
#vals = torch.mul(torch.ones((spin_n, spin_n), device=device), J)  # (this is constant)
Jij_U = vals * nonzero
Jij = Jij_U + torch.transpose(Jij_U, 0, 1)  # should always be symmetric

# max time and number of time steps
Jtmax = 5.0
Nt = 50

# dev = qml.device('lightning.gpu', wires=spin_n)
dev = qml.device('default.qubit', wires=spin_n)


@qml.qnode(dev)
def evolve_frag(state, spin_n, spin_range, frag_and_aux_list, frag_range, frag_N_list, dt, Jij, hi, model='Ising'):
    qml.QubitStateVector(state, wires=range(spin_n))
    hf.frag_circ_qlink(spin_range, frag_and_aux_list, frag_range, frag_N_list, dt, Jij, hi, model=model)
    return qml.state()


@qml.qnode(dev)
def evolve_mf_frag(state, spin_n, spin_range, frag_and_aux_list, frag_range, frag_N_list, dt, Jij, hi, sz_vec, target_neighbors,
                   model='Ising'):
    qml.QubitStateVector(state, wires=range(spin_n))
    hf.frag_mf_circ_qlink(spin_range, frag_and_aux_list, frag_range, frag_N_list, dt, Jij, hi, sz_vec, target_neighbors, model=model)
    return qml.state()


@qml.qnode(dev)
def get_op_expval(op, state, spin_n):
    qml.QubitStateVector(state, wires=range(spin_n))
    return qml.expval(op)


@qml.qnode(dev)
def get_op_expval_dt(op, spin_range, frag_and_aux, frag_range, frag_N_list, a, f, frag_list, dt, Jij, hi, binary_vec, model='Ising'):
    # this is used if NoUpdate (because you want to evolve by one time step before calculating variance in that case)
    frag_and_aux_trial, _ = torch.sort(torch.cat((frag_and_aux, torch.tensor([a], device=device))))
    frag_and_aux_trial_list = torch.nested.nested_tensor(
        [frag_and_aux_trial if i == f else frag_list[i] for i in frag_range], device=device)
    hf.initial_state(frag_and_aux_trial, binary_vec)
    hf.frag_circ_qlink(spin_range, frag_and_aux_trial_list, frag_range, frag_N_list, dt, Jij, hi, model=model)
    return qml.expval(op)


def get_rand_aux(spin_range, num_frag, frag_list, aux_list):
    # if you want random auxiliaries
    frag_and_aux_list = frag_list.clone()
    for i, frag in enumerate(frag_list):
        if i > 0 and i < num_frag - 1:
            aux_n = 2 * aux_list[i]
        else:
            aux_n = aux_list[i]
        frag_and_aux = frag_and_aux_list[i]
        uniques, counts = torch.unique(torch.cat((spin_range, frag_and_aux)), dim=0, return_counts=True)
        potential_aux = uniques[counts == 1]
        best_a_ind = torch.randperm(len(potential_aux))[:aux_n]
        best_a = potential_aux[best_a_ind]
        sorted, _ = torch.sort(torch.cat((frag_and_aux, best_a)))
        frag_and_aux_list = torch.nested.nested_tensor(
            [sorted if x == i else frag_and_aux_list[x] for x in frag_range], device=device)
    return frag_and_aux_list


def get_aux(state, spin_n, spin_range, num_frag, frag_list, aux_list, Jij):
    # gets aux according to minimum variance
    frag_and_aux_list = frag_list.clone()
    for i, frag in enumerate(frag_list):
        if i > 0 and i < num_frag - 1:
            aux_n = 2 * aux_list[i]
        else:
            aux_n = aux_list[i]
        frag_and_aux = frag_and_aux_list[i]
        uniques, counts = torch.unique(torch.cat((spin_range, frag_and_aux)), dim=0, return_counts=True)
        potential_aux = uniques[counts == 1]
        var_list = torch.zeros_like(potential_aux, dtype=torch.float64)
        for a_i, a in enumerate(potential_aux):
            op_sq, op = hf.ising_variance_op(Jij, frag_and_aux, a)
            op_exp = get_op_expval(op, state, spin_n)
            op_sq_exp = get_op_expval(op_sq, state, spin_n)
            var = op_sq_exp - op_exp ** 2
            var_list[a_i] = var
        _, indices = torch.sort(var_list, descending=True)
        best_a = potential_aux[indices[:aux_n]]
        sorted, _ = torch.sort(torch.cat((frag_and_aux, best_a)))
        frag_and_aux_list = torch.nested.nested_tensor(
            [sorted if x == i else frag_and_aux_list[x] for x in frag_range], device=device)
    return frag_and_aux_list


def get_frozen_aux(binary_vec, spin_range, num_frag, frag_list, aux_list, Jij, hi, dt, model='Ising'):
    # this is used if NoUpdate (because you want to evolve by one time step before calculating variance in that case)
    frag_and_aux_list = frag_list.clone()
    for i, frag in enumerate(frag_list):
        if i > 0 and i < num_frag - 1:
            aux_n = 2 * aux_list[i]
        else:
            aux_n = aux_list[i]

        frag_and_aux = frag_and_aux_list[i]
        uniques, counts = torch.unique(torch.cat((spin_range, frag_and_aux)), dim=0, return_counts=True)
        potential_aux = uniques[counts == 1]
        var_list = torch.zeros_like(potential_aux, dtype=torch.float64)
        for a_i, a in enumerate(potential_aux):
            op_sq, op = hf.ising_variance_op(Jij, frag_and_aux, a)
            op_exp = get_op_expval_dt(op, spin_range, frag_and_aux, frag_range, frag_N_list, a, i, frag_list, dt, Jij,
                                      hi, binary_vec, model=model)
            op_sq_exp = get_op_expval_dt(op_sq, spin_range, frag_and_aux, frag_range, frag_N_list, a, i, frag_list, dt,
                                         Jij, hi, binary_vec, model=model)
            var = op_sq_exp - op_exp ** 2
            var_list[a_i] = var
        _, indices = torch.sort(var_list, descending=True)
        best_a = potential_aux[indices[:aux_n]]
        sorted, _ = torch.sort(torch.cat((frag_and_aux, best_a)))
        frag_and_aux_list = torch.nested.nested_tensor(
            [sorted if x == i else frag_and_aux_list[x] for x in frag_range], device=device)
    return frag_and_aux_list


@qml.qnode(dev)
def get_initial_state(frag_and_aux, binary_vec):
    hf.initial_state(frag_and_aux, binary_vec)
    return qml.state()


@qml.qnode(dev)
def sz_update(state, spin_n, spin_range):
    qml.QubitStateVector(state, wires=range(spin_n))
    return [qml.expval((1 / 2) * qml.PauliZ(x.item())) for x in spin_range]


@qml.qnode(dev)
def te_rho(state, spin_n, frag_only):
    qml.QubitStateVector(state, wires=range(spin_n))
    return qml.density_matrix(frag_only.tolist())  # this is incompatible with GPU (as far as I can tell)


def get_frag_info(state, spin_n, spin_range, num_frag, frag_range, frag_list, aux_list, frag_N_list, Jij):
    frag_and_aux_list = get_aux(state, spin_n, spin_range, num_frag, frag_list, aux_list, Jij)
    target_neighbors = hf.get_mf_neighbors(spin_n, spin_range, frag_and_aux_list, frag_range, frag_N_list)
    return frag_and_aux_list, target_neighbors

def get_rand_frag_info(spin_n, spin_range, num_frag, frag_range, frag_list, aux_list, frag_N_list):
    frag_and_aux_list = get_rand_aux(spin_range, num_frag, frag_list, aux_list)
    target_neighbors = hf.get_mf_neighbors(spin_n, spin_range, frag_and_aux_list, frag_range, frag_N_list)
    return frag_and_aux_list, target_neighbors


def get_frozen_frag_info(binary_vec, spin_n, spin_range, num_frag, frag_list, aux_list, frag_N_list, hi, Jij, dt, model="Ising"):
    frag_and_aux_list = get_frozen_aux(binary_vec, spin_range, num_frag, frag_list, aux_list, Jij, hi, dt, model=model)
    target_neighbors = hf.get_mf_neighbors(spin_n, spin_range, frag_and_aux_list, frag_range, frag_N_list)
    return frag_and_aux_list, target_neighbors


def get_frag_fid(spin_n, spin_range, num_frag, frag_range, frag_and_aux_list, frag_list, aux_list, frag_N_list, dJt, dt, Nt, Jij, hi,
                 target_neighbors, binary_vec, MF=True, Update=True, Rand=False, model='Ising'):
    N_half = range(int(np.floor(spin_n / 2)))
    spin_range_as_frag = torch.nested.nested_tensor([spin_range], device=device)
    one_frag_range = torch.tensor([0], device=device)
    spin_n_as_tensor = torch.tensor([spin_n], device=device)
    fid_vec = torch.zeros((num_frag, Nt+1), device=device)
    fid_full = torch.zeros(Nt+1, device=device)
    ent_vec = torch.zeros(Nt+1, device=device)
    frag_state = get_initial_state(spin_range, binary_vec)
    ex_state = get_initial_state(spin_range, binary_vec)
    for f, frag in enumerate(frag_list):
        frag_rho = te_rho(frag_state, spin_n, frag)
        ex_rho = te_rho(ex_state, spin_n, frag)
        fid_vec[f, 0] = qml.math.fidelity(torch.tensor(frag_rho), torch.tensor(ex_rho)).item()
    frag_rho = te_rho(frag_state, spin_n, spin_range)
    ex_rho = te_rho(ex_state, spin_n, spin_range)
    fid_full[0] = qml.math.fidelity(torch.tensor(frag_rho), torch.tensor(ex_rho)).item()

    for N in range(Nt):
        Jt = (N + 1) * dJt

        # Evolve full state
        ex_state = evolve_frag(ex_state, spin_n, spin_range, spin_range_as_frag, one_frag_range, spin_n_as_tensor, dt, Jij, hi, model=model)
        ent_vec[N+1] = qml.math.vn_entropy(ex_state, N_half)

        # next, evolve frags to next state and calculate fidelity
        if MF:
            # update expectation values
            sz_vec = sz_update(frag_state, spin_n, spin_range)
            frag_state = evolve_mf_frag(frag_state, spin_n, spin_range, frag_and_aux_list, frag_range, frag_N_list, dt,
                                        Jij, hi, sz_vec, target_neighbors, model=model)

        else:
            frag_state = evolve_frag(frag_state, spin_n, spin_range, frag_and_aux_list, frag_range, frag_N_list, dt, Jij, hi, model=model)

        for f, frag in enumerate(frag_list):
            frag_rho = te_rho(frag_state, spin_n, frag)
            ex_rho = te_rho(ex_state, spin_n, frag)
            fid_vec[f, N+1] = qml.math.fidelity(torch.tensor(frag_rho), torch.tensor(ex_rho)).item()

        frag_rho = te_rho(frag_state, spin_n, spin_range)
        ex_rho = te_rho(ex_state, spin_n, spin_range)
        fid_full[N+1] = qml.math.fidelity(torch.tensor(frag_rho), torch.tensor(ex_rho)).item()

        # Update auxiliary encodings (unless NoUpdate)
        if Update:
            if Rand:
                frag_and_aux_list, target_neighbors = get_rand_frag_info(spin_n, spin_range, num_frag, frag_range,
                                                                         frag_list, aux_list, frag_N_list)

            else:
                frag_and_aux_list, target_neighbors = get_frag_info(frag_state, spin_n, spin_range, num_frag,
                                                                    frag_range, frag_list, aux_list, frag_N_list, Jij)

    return fid_vec, fid_full, ent_vec

H_dim = (1 << spin_n)
psi0 = int(torch.randint(H_dim, (1,)))
binary_vec = hf.get_binary_state(psi0, spin_n)

dJt = Jtmax / Nt
Jt_vec = torch.arange(dJt, Jtmax + dJt, dJt)
dt = dJt / torch.max(torch.abs(Jij))

model = 'Ising'  # always at this point
frag_state = get_initial_state(spin_range, binary_vec)


# these are the auxiliary encoding settings: if Rand, the encoding is random (otherwise, obeys variance rule).
# if Update, the encoding is updated each time step
Rand = False
Update = True

if Rand:
    frag_and_aux_list, target_neighbors = get_rand_frag_info(spin_n, spin_range, num_frag, frag_range,
                                                            frag_list, aux_list, frag_N_list)
else:
    if Update:
        frag_and_aux_list, target_neighbors = get_frag_info(frag_state, spin_n, spin_range, num_frag, frag_range,
                                                            frag_list, aux_list, frag_N_list, Jij)
    else:
        frag_and_aux_list, target_neighbors = get_frozen_frag_info(binary_vec, spin_n, spin_range, num_frag, frag_list,
                                                                   aux_list, frag_N_list, hi, Jij, dt, model=model)

# all that I extract is the various fidelities (retrieve fragment i with fid_mf_vec[i,:]), and the bipartite entanglement of the exact simulation
# mean-field corrected:
fid_mf_vec, fid_mf_full, ent_vec = get_frag_fid(spin_n, spin_range, num_frag, frag_range, frag_and_aux_list, frag_list,
                                                aux_list, frag_N_list, dJt, dt, Nt, Jij, hi, target_neighbors,
                                                binary_vec, MF=True, Update=Update, Rand=Rand, model=model)

# no mean-field:
fid_vec, fid_full, ent_vec = get_frag_fid(spin_n, spin_range, num_frag, frag_range, frag_and_aux_list, frag_list,
                                          aux_list, frag_N_list, dJt, dt, Nt, Jij, hi, target_neighbors, binary_vec,
                                          MF=False, Update=Update, Rand=Rand, model=model)

print(fid_mf_vec[0, :])
