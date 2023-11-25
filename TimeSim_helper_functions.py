
import torch
import pennylane as qml

#dtype = torch.complex128
dtype = torch.complex64
device = torch.device("cpu")

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

def get_binary_state(k, spin_n):
    if (k > (2 ** spin_n) - 1) or (k < 0):
        print(str('Invalid basis state for ' + str(spin_n) + ' qubits.'))
    else:
        bit_string = f'{k:0{spin_n}b}'
        binary_state = torch.tensor([int(bit) for bit in bit_string], device=device)
        return binary_state


def get_active_interactions(frag_and_aux_list, frag_range, frag_N_list):
    active_combinations = get_interactions(frag_and_aux_list[0], frag_N_list[0])
    for i in frag_range[1:]:
        combinations = get_interactions(frag_and_aux_list[i], frag_N_list[i])
        active_combinations = torch.cat((active_combinations, combinations))
    return torch.unique(active_combinations, dim=0)

def get_mf_neighbors(spin_n, spin_range, frag_and_aux_list, frag_range, frag_N_list):
    active_combinations = get_active_interactions(frag_and_aux_list, frag_range, frag_N_list)
    total_combinations = get_interactions(spin_range, spin_n)
    uniques, counts = torch.unique(torch.cat((active_combinations, total_combinations)), dim=0, return_counts=True)
    mf_combinations = uniques[counts == 1]
    target_neighbors = []
    for spin in spin_range:
        neighbors = torch.empty(0, device=device)
        for combo in mf_combinations:
            if spin in combo:
                neighbors = torch.cat((neighbors, combo[torch.remainder((combo == spin).nonzero()[0] + 1, 2)]))
        target_neighbors.append(neighbors.int())

    return torch.nested.nested_tensor(target_neighbors, device=device)

def initial_state(frag_and_aux, binary_vec):
    for ind, spin in enumerate(binary_vec):
        if spin == 1 and ind in frag_and_aux:
            qml.PauliX(ind)

def model_ham_qlink(spin_range, frag_and_aux_list, frag_range, frag_N_list, Jij, hi, model='Ising'):
    coeffs = []
    obs = []
    combinations = get_active_interactions(frag_and_aux_list, frag_range, frag_N_list)
    if model == 'Ising':
        for combo in combinations:
            coeffs.append(-Jij[combo[0], combo[1]]/4)
            obs.append(qml.PauliZ(combo[0].item()) @ qml.PauliZ(combo[1].item()))
        for spin in spin_range:
            coeffs.append(-hi[spin]/2)
            obs.append(qml.PauliX(spin.item()))
    H = qml.Hamiltonian(coeffs, obs)
    return H, coeffs, obs

def model_ham_clink(frag_and_aux, frag_N, Jij, hi, model='Ising'):
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

def get_ising_MF_correction_terms_qml(Jij, z_mag_r, frag_and_aux, target_neighbors):
    coeffs = []
    obs = []
    for i, ind in enumerate(frag_and_aux):
        b = 0
        for j in target_neighbors[i]:
            b -= Jij[ind, j] * z_mag_r[j]

        coeffs.append(b / 2)
        obs.append(qml.PauliZ(ind.item()))
    return coeffs, obs

def frag_mf_circ_qlink(spin_range, frag_and_aux_list, frag_range, frag_N_list, dt, Jij, hi, sz_vec, target_neighbors, model='Ising'):
    sub_H, coeffs, obs = model_ham_qlink(spin_range, frag_and_aux_list, frag_range, frag_N_list, Jij, hi, model=model)
    correction_coeffs, correction_obs = get_ising_MF_correction_terms_qml(Jij, sz_vec, spin_range, target_neighbors)
    sub_H_MF = qml.Hamiltonian(coeffs + correction_coeffs, obs + correction_obs)
    qml.exp(sub_H_MF, -1j * dt)  # this is incompatible with GPU (as far as I can tell)

def frag_circ_qlink(spin_range, frag_and_aux_list, frag_range, frag_N_list, dt, Jij, hi, model='Ising'):
    sub_H, coeffs, obs = model_ham_qlink(spin_range, frag_and_aux_list, frag_range, frag_N_list, Jij, hi, model=model)
    qml.exp(sub_H, -1j * dt)  # this is incompatible with GPU (as far as I can tell)

def ising_variance_op(Jij, frag_and_aux, a):
    # a is the potential auxiliary, frag_and_aux includes all qubits in the fragment
    coeffs_sq = []
    ops_sq = []
    coeffs = []
    ops = []
    for f in frag_and_aux:
        coeffs.append(Jij[a, f] / 4)
        ops.append(qml.PauliZ(a.item()) @ qml.PauliZ(f.item()))
        for f_p in frag_and_aux:
            coeffs_sq.append((Jij[a, f] / 4) * (Jij[a, f_p] / 4))
            if f == f_p:
                ops_sq.append(qml.Identity(a.item()) @ qml.Identity(f.item()))
            else:
                ops_sq.append(qml.Identity(a.item()) @ qml.PauliZ(f.item()) @ qml.PauliZ(f_p.item()))
    op = qml.Hamiltonian(coeffs, ops)
    op_sq = qml.Hamiltonian(coeffs_sq, ops_sq)
    return op_sq, op


