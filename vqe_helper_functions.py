
import torch
import numpy as np
import pennylane as qml
from scipy import sparse

#dtype = torch.complex128
dtype = torch.complex64
device = torch.device("cpu")

Sx = torch.tensor([[0,   0.5], [0.5,  0]], device=device, dtype=dtype)
Sy = torch.tensor([[0, -0.5j], [0.5j, 0]], device=device, dtype=dtype)
Sz = torch.tensor([[0.5,   0], [0, -0.5]], device=device, dtype=dtype)
Id = torch.tensor([[1,     0], [0,    1]], device=device, dtype=dtype)

# the batch of functions below is really just to create a (sparse) matrix of the
# full Hamiltonian so that we can compute the exact g.s. energy using exact diagonalization
# Unfortunately, this requires scipy -- no method currently in pytorch to efficiently compute smallest eigenvalue of a sparse matrix
def get_op(op_i):
    if op_i == 0:
        op = Id
    elif op_i == 1:
        op = Sx
    elif op_i == 2:
        op = Sy
    elif op_i == 3:
        op = Sz
    return op

def get_sparse_op(op_i):
    op = sparse.csr_matrix(get_op(op_i))
    return op

def get_sparse_op_mat(op):
    for i in range(len(op)):
        spin_ind = 0
        if i == 0:
            op_mat = get_sparse_op(op[i])
            spin_ind += 1

        else:
            next_op = get_sparse_op(op[i])
            spin_ind += 1
            op_mat = sparse.kron(op_mat, next_op)

    return op_mat

def create_sparse_ising_ham(Jij, hi, spin_n, spin_ind):
    Jij = np.array(Jij)
    hi = np.array(hi)
    for i in spin_ind:
        op = torch.tensor([0 for _ in np.arange(i)] + [1] + [0 for _ in np.arange(spin_n - i - 1)], device=device)
        op_mat = get_sparse_op_mat(op)
        if i > 0:
            Ham -= hi[i]*op_mat
        else:
            Ham = hi[i]*op_mat
        for j in torch.arange(i):
            op = torch.tensor([0 for _ in torch.arange(j)] + [3] + [0 for _ in torch.arange(i - j - 1)] + [3] + [0 for _ in torch.arange(spin_n - i - 1)], device=device)
            op_mat = get_sparse_op_mat(op)
            Ham -= Jij[i, j]*op_mat
    return Ham

def gen_partition(spin_n, M, a):
    Mp1 = M + 1
    # generate random partition of the circuit, with sub-circuit partitions of max size M + a
    frag_partition = torch.randint(1, Mp1, (1,), device=device)
    while torch.sum(frag_partition) < spin_n:
        frag_partition = torch.cat((frag_partition, torch.randint(1, Mp1, (1,), device=device)))
    frag_partition[-1] = spin_n - torch.sum(frag_partition[:-1])
    num_frag = frag_partition.size(dim=0)
    frag_range = torch.arange(num_frag, device=device)
    frag_partition = frag_partition[torch.randperm(num_frag, device=device)]
    frag_list = torch.nested.nested_tensor([torch.arange(frag_partition[x], device=device) + torch.sum(frag_partition[:x])
                                            for x in frag_range], device=device)
    aux_list = torch.tensor([int(a/2) for _ in frag_range], device=device)
    aux_list[0] = a
    aux_list[-1] = a

    return frag_list, aux_list, num_frag, frag_range

def get_interactions(frag_and_aux, frag_N):
    combinations = torch.zeros((int(frag_N * (frag_N - 1)/2), 2), dtype=torch.int64, device=device)
    ind = 0
    combo = 0
    for i in frag_and_aux:
        ind += 1
        for j in frag_and_aux[ind:]:
            combinations[combo, :] = torch.tensor([i, j], device=device)
            combo += 1
    return combinations

def get_linear_circuit(params_frag, rots, layers, frag_and_aux, ent_gate=1):
    param_index = 0
    for rot in rots:
        for qubit in frag_and_aux:
            if rot == 3:
                qml.RZ(params_frag[param_index], wires=qubit.item())
            elif rot == 2:
                qml.RY(params_frag[param_index], wires=qubit.item())
            elif rot == 1:
                qml.RX(params_frag[param_index], wires=qubit.item())
            param_index += 1

    for l in np.arange(layers):
        for qubit in frag_and_aux[:-1]:
            if np.mod(qubit, 2) == 0:
                if ent_gate == 1:
                    qml.CNOT(wires=[qubit.item(), qubit.item()+1])
                elif ent_gate == 2:
                    qml.CY(wires=[qubit.item(), qubit.item()+1])
                elif ent_gate == 3:
                    qml.CZ(wires=[qubit.item(), qubit.item()+1])

        for qubit in frag_and_aux[:-1]:
            if np.mod(qubit, 2) == 1:
                if ent_gate == 1:
                    qml.CNOT(wires=[qubit.item(), qubit.item()+1])
                elif ent_gate == 2:
                    qml.CY(wires=[qubit.item(), qubit.item()+1])
                elif ent_gate == 3:
                    qml.CZ(wires=[qubit.item(), qubit.item()+1])

        for rot in rots:
            for qubit in frag_and_aux:
                if rot == 3:
                    qml.RZ(params_frag[param_index], wires=qubit.item())
                elif rot == 2:
                    qml.RY(params_frag[param_index], wires=qubit.item())
                elif rot == 1:
                    qml.RX(params_frag[param_index], wires=qubit.item())
                param_index += 1


# need's doing: can we eliminate lists?
def model_ham(frag_and_aux, frag_N, Jij, hi, model='Ising'):
    coeffs = []
    obs = []
    combinations = get_interactions(frag_and_aux, frag_N)
    if model == 'Ising':
        for combo in combinations:
            coeffs.append(-Jij[combo[0], combo[1]]/4)
            obs.append(qml.PauliZ(combo[0].item()) @ qml.PauliZ(combo[1].item()))
        for spin in frag_and_aux:
            coeffs.append(-hi[spin]/2)
            obs.append(qml.PauliX(spin.item()))
    H = qml.Hamiltonian(coeffs, obs)
    return H, coeffs, obs

def get_frag_info_circ(spin_n, spin_ind, frag_list, aux_list, num_frag):
    # Just gathers all the important info for this particular fragmentation
    frag_sys_N_list = torch.tensor([frag_list[x].size(dim=0) for x in torch.arange(num_frag)], device=device)
    frag_N_list = torch.tensor([frag_sys_N_list[x] + aux_list[x] if x == 0 or x == num_frag - 1
                                else frag_sys_N_list[x] + int(2 * aux_list[x]) for x in torch.arange(num_frag)],
                               device=device)

    # these will be nested tensors
    frag_and_aux_list = [torch.zeros(frag_N_list[x], dtype=torch.int64, device=device) for x in torch.arange(num_frag)]
    aux_target_list = [torch.zeros(frag_N_list[x] - frag_sys_N_list[x], dtype=torch.int64, device=device) for x in
                       torch.arange(num_frag)]
    frag_env_list = [torch.zeros(spin_n - frag_N_list[x], dtype=torch.int64, device=device) for x in torch.arange(num_frag)]

    for i, frag in enumerate(frag_list):
        aux_target = aux_target_list[i]
        a_ind = 0
        for a in torch.arange(aux_list[i], device=device):
            if i == 0:
                aux_target[a_ind] = int(frag[-1] + (a + 1))
                a_ind += 1

            elif i == num_frag - 1:
                aux_target[a_ind] = int(frag[0] - (a + 1))
                a_ind += 1

            else:
                aux_target[a_ind] = int(frag[0] - (a + 1))
                a_ind += 1
                aux_target[a_ind] = int(frag[-1] + (a + 1))
                a_ind += 1

        frag_and_aux, indices = torch.sort(torch.cat((frag_list[i], aux_target)))
        frag_and_aux_list[i] += frag_and_aux
        aux_target_list[i] += aux_target

    frag_and_aux_list = torch.nested.nested_tensor(frag_and_aux_list, device=device)
    aux_target_list = torch.nested.nested_tensor(aux_target_list, device=device)

    for i, frag_and_aux in enumerate(frag_and_aux_list):
        combined = torch.cat((frag_and_aux, spin_ind))
        uniques, counts = combined.unique(return_counts=True)
        frag_env_list[i] += uniques[counts == 1]

    frag_env_list = torch.nested.nested_tensor(frag_env_list, device=device)

    return frag_N_list, frag_sys_N_list, aux_target_list, frag_and_aux_list, frag_env_list


def get_sub_H_info(frag_and_aux_list, frag_N_list, frag_range, Jij, hi, model='Ising'):
    sub_Ham_list = []
    coeff_list = []
    obs_list = []
    for i in frag_range:
        frag_and_aux = frag_and_aux_list[i]
        frag_N = frag_N_list[i]
        sub_Ham, coeffs, obs = model_ham(frag_and_aux, frag_N, Jij, hi, model=model)
        sub_Ham_list.append(sub_Ham)
        coeff_list.append(coeffs)
        obs_list.append(obs)

    return sub_Ham_list, coeff_list, obs_list

# might need to tweak this one
def get_ising_MF_correction_terms_qml(Jij, z_mag_r, aux_ind, target_ind, target_neighbors):
    # aux_ind is the index of the auxiliary w/in the list of auxiliaries for this fragment
    # getting mf correction operators / terms
    coeffs = []
    obs = []
    for i in torch.arange(aux_ind.size(dim=0)):
        ind = target_ind[i]
        b = 0
        for j in torch.arange(target_neighbors.size(dim=0)):
            k = target_neighbors[j]
            b -= Jij[target_ind[i], k] * z_mag_r[k]

        coeffs.append(b / 2)
        obs.append(qml.PauliZ(ind.item()))

    return coeffs, obs



