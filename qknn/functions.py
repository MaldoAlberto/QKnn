"""
Various functions needed to implement the Q-kNN.
"""
import math
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister, transpile
# TODO: enforce PEP-8 style, naming conventions, imports, etc.


def indexPosicion(value:int = 0):
    """
    Convert an integer value into binary representation, then flip the digits
    into Qiskit qubit ordering convention. So, e.g.,

    4 -> 0b100 -> |q0 q1 q2> -> [0,0,1] -> returns [2]
    
    Args:
        value: integer value that we transform in binary number that representate in a list position
    
    Returns:
        list: a integer list of position that representate the index position oof ones in a binary number 
    
    """
    # TODO: explain args, add type signatures
    list_bin = []
    temp = bin(value)[2:]
    temp = temp[::-1]
    for i, v in enumerate(temp):
        if v == '1':
            list_bin.append(i)
    return list_bin


def diffuser(nqubits:int = 2):
    """
    Construct the Grover diffuser gate for `nqubits` qubits.
    
    
    Args:
        nqubits: integer value that indicate the number of qubits to build the diffuser gate
    
    Returns:
        qiskit.circuit.gate.Gate: a quantum gate with the label of diffuser 
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


def qram(size_QRAM:int = 1,features:int = 1,train_set:list = None):
    """
    Build a QRAM with CRY or H-CRZ gates.
    
    Args:
        size_QRAM: integer value that indicate the address qubits
        features: integer value  of the number of rotations gates to save in the QRAM
        train_set: list of the train set
    
    Returns:
        qiskit.circuit.quantumcircuit.QuantumCircuit: a QRAM that has in RY or RZ the information of a dataset
    """
    # TODO: explain args, add type signatures
    # TODO: pass in the rotation gate as a function, ideally, or as a string as fallback
    address = QuantumRegister(size_QRAM,name = "address qubits")
    ancilla = QuantumRegister(1, name = "ancilla qubit")
    data = QuantumRegister(features,name = "train data qubits")
    

    qc = QuantumCircuit(address,ancilla,data)
    
    len_arr = (2**size_QRAM)
    
    # cap array length to be the length of the trainings set
    if len_arr > len(train_set):
        len_arr = len(train_set)
        
        
    for i in range(len_arr):
        x_gates_array = indexPosicion(i)
        if x_gates_array:
            qc.x(address[x_gates_array])
            qc.mcx(address,ancilla)
            angles = train_set[i]
            for index in range(len(angles)):
                qc.cry(angles[index],ancilla, data[index])
            qc.mcx(address,ancilla)
            qc.x(address[x_gates_array])
        else:
            qc.mcx(address,ancilla)
            angles = train_set[i]
            for index in range(len(angles)):
                qc.cry(angles[index],ancilla, data[index])
            qc.mcx(address,ancilla)
 
        qc.barrier()
    return qc
    # TODO: rewrite in functional form...


#def qram_d(qc, address, ancilla, data, train_set, len_arr):
#    """
#    Run the `qram` function with a different default (placeholder).
#    """
#    # TODO: explain args, add type signatures... well, or get rid of this function in notebooks
#    return qram(qc, address, ancilla, data, train_set, len_arr, dagger=True)

def oracle_st(features:int,test_value:list):
    """
    Build a Oracle using SWAP-Test.
    
    Args:
        features: integer value  of the number of rotations gates to comparate in the SWAP-Test
        test_value: list of an instance of the test set
    
    Returns:
        qiskit.circuit.quantumcircuit.QuantumCircuit: a SWAP-Test with a mid measurement to 
        comparate with two state of size features.
    """
    
    
    data_train = QuantumRegister(features,name = "train data qubits")
    data_test = QuantumRegister(features,name = "test data qubits")
    swap_test = QuantumRegister(1, name = "SWAP-Test qubit")
    oracle = QuantumRegister(1, name = "oracle qubit")
    c_oracle = ClassicalRegister(1, name = "mid measure")
    qc = QuantumCircuit(data_train,data_test,swap_test,oracle,c_oracle)
    
    
    qc.h(swap_test)
    for i in range(features):
        qc.ry(test_value[i],data_test[i])
        qc.cswap(swap_test,data_train[i],data_test[i])
    qc.h(swap_test)
    qc.barrier()
    
    qc.measure(swap_test,c_oracle)
    qc.x(oracle).c_if(c_oracle[0], 0)
    qc.barrier()
    
    
    for i in range(features):
        qc.ry(-test_value[i],data_test[i])    
    qc.barrier()
    
    return qc



def qknn(test_value:list, train_set:list,size_QRAM,features,max_trials:int=1):
    """
    Build a QKNN.
    
    Args:
        test_value: list of an instance of the test set
        train_set: list of the train set
        size_QRAM: integer value that indicate the address qubits
        features: integer value  of the number of rotations gates to save in the QRAM
        max_trials: integer number to repeat the oracle and diffuser follows the Grover's algorithm format
    
    Returns:
        qiskit.circuit.quantumcircuit.QuantumCircuit: a QKNN circuit for a comparition of
        a particular test set instance with a sample or all the train set
    """
    # TODO: explain args, add type signatures
    # TODO: add a flag to choose rotatio gates
    # TODO: remove unused arguments
    # TODO: fix for "arbitrary" (somewhat) input data - at least remove as much hardcoding as possible
    n = (2**size_QRAM)
    n_grover_trials = math.ceil(math.sqrt(n))
    if max_trials:
        n_grover_trials = min(max_trials, n_grover_trials)
    
    
    address = QuantumRegister(size_QRAM,name = "address qubits")
    ancilla = QuantumRegister(1,name = "ancilla qubits")
    data_train = QuantumRegister(features,name = "train data qubits")
    data_test = QuantumRegister(features,name = "test data qubits")
    swap_test = QuantumRegister(1,name = "SWAP-Test qubit")
    oracle = QuantumRegister(1,name = "Oracle qubit")
    c = ClassicalRegister(size_QRAM,name = "address bits")
    c_oracle = ClassicalRegister(1,name = "oracle bit")
    qc = QuantumCircuit(address, ancilla, data_train, data_test, swap_test,oracle, c,c_oracle)

    qc.h(address)
    qc.x(oracle)
    qc.h(oracle)
    
    qc.barrier()
    
    
    for _ in range(n_grover_trials):

        qc.append(qram(size_QRAM,features,train_set),address[:] + ancilla[:] + data_train[:])
        
        qc.append(oracle_st(features,test_value), data_train[:] + data_test[:] + swap_test[:] + oracle[:], c_oracle[:])

        qc.append(qram(size_QRAM,features,train_set).inverse(),address[:] + ancilla[:] + data_train[:])
        
        qc.append(diffuser(size_QRAM),address)
        qc.barrier() 

    qc.x(address)
    qc.measure(address,c)
    
    return qc
