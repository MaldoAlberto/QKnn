"""
Various functions needed to implement the Q-kNN.
"""
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister  #, transpile
# TODO: enforce PEP-8 style, naming conventions, imports, etc.


def index_positions(value):
    """
    Convert an integer value into binary representation, then flip the digits
    into Qiskit qubit ordering convention. So, e.g.,

    4 -> 0b100 -> |q0 q1 q2> -> [0,0,1] -> returns [2]
    """
    # TODO: explain args, add type signatures
    list_bin = []
    temp = bin(value)[2:]
    temp = temp[::-1]
    for i, v in enumerate(temp):
        if v == '1':
            list_bin.append(i)
    return list_bin


def diffuser(nqubits):
    """
    Construct the Grover diffuser gate for `nqubits` qubits.
    """
    # TODO: explain args, add type signatures
    qc = QuantumCircuit(nqubits)
    for qubit in range(nqubits):
        qc.h(qubit)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mcx(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    for qubit in range(nqubits):
        qc.x(qubit)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Return the diffuser as a gate
    u_s = qc.to_gate()
    u_s.name = "$U_s$"
    return u_s


def qram(qc, address, ancilla, data, train_set, len_arr, dagger=False):
    """
    Modify the QuantumCircuit `qc` in-place to fill the QRAM with data.
    """
    # TODO: explain args, add type signatures
    # TODO: pass in the rotation gate as a function, ideally, or as a string as fallback
    sign_val = -1 if dagger else 1
    # cap array length to be the length of the trainings set
    if len_arr > len(train_set):
        len_arr = len(train_set)
    #
    for i in range(len_arr):
        x_gates_array = index_positions(i)
        if x_gates_array:
            qc.x(address[x_gates_array])
            qc.mcx(address, ancilla)
            angles = train_set[i]
            for index in range(len(angles)):
                qc.crz(sign_val * angles[index], ancilla, data[index])
            qc.mcx(address, ancilla)
            qc.x(address[x_gates_array])
        else:
            qc.mcx(address, ancilla)
            angles = train_set[i]
            for index in range(len(angles)):
                qc.crz(sign_val * angles[index],ancilla, data[index])
            qc.mcx(address,ancilla)
        qc.barrier()
    # TODO: rewrite in functional form...


def qram_d(qc, address, ancilla, data, train_set, len_arr):
    """
    Run the `qram` function with a different default (placeholder).
    """
    # TODO: explain args, add type signatures... well, or get rid of this function in notebooks
    return qram(qc, address, ancilla, data, train_set, len_arr, dagger=True)


def qknn(input_a, train_set, y_train, size, max_trials=None):
    """
    Run the Q-kNN
    """
    # TODO: explain args, add type signatures
    # TODO: add a flag to choose rotatio gates
    # TODO: remove unused arguments
    # TODO: fix for "arbitrary" (somewhat) input data - at least remove as much hardcoding as possible
    n = (2**size)
    n_grover_trials = math.ceil(math.sqrt(n))
    if max_trials:
        n_grover_trials = min(max_trials, n_grover_trials)

    address = QuantumRegister(size)
    ancilla = QuantumRegister(1)
    data_train = QuantumRegister(4)
    data_test = QuantumRegister(4)
    swap_test = QuantumRegister(1)
    oracle = QuantumRegister(1)
    c = ClassicalRegister(size)
    c_oracle = ClassicalRegister(1)
    qc = QuantumCircuit(address, ancilla, data_train, data_test, swap_test,oracle, c,c_oracle)

    qc.h(address)
    qc.x(oracle)
    qc.h(oracle)
    qc.h(swap_test)
    qc.h(data_train)
    qc.barrier()

    for _ in range(n_grover_trials):
        qram(qc,address,ancilla,data_train,train_set,n)
        qc.barrier()
        angles = input_a

        # test train
        for i, v in enumerate(angles):
            qc.h(data_test[i])
            qc.rz(v, data_test[i])
        qc.barrier()

        # SWAP-Test
        for i in range(4):    # TODO: hard-coded number of features
            qc.cswap(swap_test[0], data_train[i], data_test[i])
        qc.h(swap_test)
        qc.barrier()

        qc.measure(swap_test, c_oracle)
        qc.x(oracle).c_if(c_oracle[0], 0)

        qc.barrier()
        qram_d(qc,address,ancilla,data_train,train_set,n)
        qc.barrier()
        qc.append(diffuser(size),address)
        qc.barrier()

    qc.x(address)
    qc.measure(address,c)

    return qc