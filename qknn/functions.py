"""
Various functions needed to implement the Q-kNN.
"""
import math
<<<<<<< HEAD
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
=======
from sklearn.metrics import accuracy_score
# import numpy for postprocessing to find the k-nn label
import numpy as np
# import qiskit minimal methods to use quantum circuit and use qubits as the classical bits
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister, transpile
from qiskit_aer import AerSimulator
>>>>>>> upstream/main
from qiskit.circuit import CircuitInstruction, Instruction
from qiskit.circuit.library.standard_gates import RYGate,RZGate


def index_positions(value:int = 0):
    """
    Convert an integer value into binary representation, then flip the digits
    into Qiskit qubit ordering convention. So, e.g.,

    4 -> 0b100 -> |q0 q1 q2> -> [0,0,1] -> returns [2]

    Args:
        value: integer value that we transform in binary number that representate in a list position

    Returns:
        list: a integer list of position that representate the index position of ones in a binary
        number
    """
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
    u_gate = qc.to_gate()
    u_gate.name = "$diffuser$"
    return u_gate


def qram(
        size_QRAM:int = 1,
        features:int = 1,
        train_set:list = [[1],[1]],
        gate:Instruction = RYGate
    ):  # pylint: disable=invalid-name,dangerous-default-value
    """
    Build a QRAM with CRY or H-CRZ gates.

    Args:
        size_QRAM: integer value that indicate the address qubits
        features: integer value  of the number of rotations gates to save in the QRAM
        train_set: list of the train set

    Returns:
        qiskit.circuit.quantumcircuit.QuantumCircuit: a QRAM that has the dataset
        encoded through RZ or RY gates.
    """
    address = QuantumRegister(size_QRAM, name = "address qubits")
    ancilla = QuantumRegister(1, name = "ancilla qubit")
    data = QuantumRegister(features, name = "train data qubits")
    qc = QuantumCircuit(address, ancilla, data)

    len_arr = 2**size_QRAM

    # cap array length to be the length of the trainings set
    if len_arr > len(train_set):
        len_arr = len(train_set)

    for i in range(len_arr):
        x_gates_array = index_positions(len_arr-i-1)
        if x_gates_array:
            qc.x(address[x_gates_array])
        qc.mcx(address,ancilla)
        angles = train_set[i]
        for index, value in enumerate(angles):
<<<<<<< HEAD
            qc._append(   # pylint: disable=protected-access
=======
            qc.append(   # pylint: disable=protected-access
>>>>>>> upstream/main
                CircuitInstruction(gate(value).control(), [ancilla[0], data[index]])
            )
        qc.mcx(address,ancilla)
        if x_gates_array:
            qc.x(address[x_gates_array])
        qc.barrier()

    return qc


<<<<<<< HEAD
def oracle_st(features:int, test_value:list, gate:Instruction=RYGate):
=======
def oracle_st(features:int=1, test_value:list=[[1]], gate:Instruction=RYGate):
>>>>>>> upstream/main
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
<<<<<<< HEAD
        qc._append(  # pylint: disable=protected-access
=======
        qc.append(  # pylint: disable=protected-access
>>>>>>> upstream/main
            CircuitInstruction(gate(test_value[i]), [data_test[i]])
        )
        qc.cswap(swap_test,data_train[i],data_test[i])
    qc.h(swap_test)
    qc.barrier()

    qc.measure(swap_test,c_oracle)
    qc.x(oracle).c_if(c_oracle[0], 0)
    qc.barrier()

    for i in range(features):
<<<<<<< HEAD
        qc._append(  # pylint: disable=protected-access
=======
        qc.append(  # pylint: disable=protected-access
>>>>>>> upstream/main
            CircuitInstruction(gate(-test_value[i]), [data_test[i]])
        )
    qc.barrier()

    return qc


def qknn(
        test_value:list=[[1],[1]],
        train_set:list=[[1],[1]],
        size_QRAM:int=1,
        features:int=1,
        max_trials:int=1,
        rotation:str="ry"
    ):  # pylint: disable=invalid-name
    """
    Build a QKNN.

    Args:
        test_value: list of an instance of the test set
        train_set: list of the train set
        size_QRAM: integer value that indicate the address qubits
        features: integer value  of the number of rotations gates to save in the QRAM
        max_trials: integer number to repeat the oracle and diffuser follows the
        Grover's algorithm format

    Returns:
        qiskit.circuit.quantumcircuit.QuantumCircuit: a QKNN circuit for a comparition of
        a particular test set instance with a sample or all the train set
    """
    n = 2**size_QRAM
    n_grover_trials = math.ceil(math.sqrt(n))
    if max_trials:
        n_grover_trials = min(max_trials, n_grover_trials)

    rotation_gates = {"ry": RYGate, "rz": RZGate}
    gate = rotation_gates[rotation]

    address = QuantumRegister(size_QRAM, name = "address qubits")
    ancilla = QuantumRegister(1, name = "ancilla qubits")
    data_train = QuantumRegister(features, name = "train data qubits")
    data_test = QuantumRegister(features, name = "test data qubits")
    swap_test = QuantumRegister(1, name = "SWAP-Test qubit")
    oracle = QuantumRegister(1, name = "Oracle qubit")
    c = ClassicalRegister(size_QRAM, name = "address bits")
    c_oracle = ClassicalRegister(1, name = "oracle bit")
    qc = QuantumCircuit(address, ancilla, data_train, data_test, swap_test,oracle, c, c_oracle)

    qc.h(address)
    qc.x(oracle)
    qc.h(oracle)

    qc.barrier()

    for _ in range(n_grover_trials):
        if gate == RZGate:
            qc.h(data_train)
            qc.h(data_test)
        qc.append(qram(size_QRAM, features, train_set, gate),
                  address[:] + ancilla[:] + data_train[:])
        qc.append(oracle_st(features,test_value,gate),
                  data_train[:] + data_test[:] + swap_test[:] + oracle[:],
                  c_oracle[:])
        qc.append(qram(size_QRAM, features, train_set, gate).inverse(),
                  address[:] + ancilla[:] + data_train[:])
        qc.append(diffuser(size_QRAM),address)
        qc.barrier()
        if gate == RZGate:
            qc.h(data_train)
            qc.h(data_test)

<<<<<<< HEAD
    qc.x(address)
    qc.measure(address, c)

    return qc
=======
    #qc.x(address)
    qc.measure(address, c)

    return qc
    
    
    
class qknn_experiments():
    """
    Class to use multiple knn experiments
    """
    
    def __init__(self,x_test:list=[[1]],
                    x_train:list=[[1]],
                    y_test:list=[[1]],
                    y_train:list=[[1]],
                    features:int=1,
                    max_trials:int=1,
                    rotation:str="ry",
                    experiment_size:int=1,
                   min_QRAM:int=3,
                    max_QRAM:int=4):
        """
    Initialize the class
        Args:
        self: is part of the class 
        x_test: list of list of float values that contain the test set
        x_train: list of list of float values that contain the train set
        y_test: list of integer values that contain the test labels 
        y_train: list of integer values that contain the train labels
        features: integer value that indicates the number of features to use in the QRAM
        max_trials: integer value that representates the number of iterations to use the quantum circuit follows the Grover's algorithm
        rotation: Quantum gate that indicates the rotation gate to use to encode
        experiment_size: integer value that indicates the number of experiments of each QRAM 
        min_QRAM: integer value to implement the minimum QRAM
        max_QRAM: integer value to implement the maximum QRAM
        """     
        self.x_test = x_test
        self.x_train = x_train
        self.y_test = y_test
        self.y_train = y_train
        self.features = features
        self.max_trials = max_trials
        self.rotation = rotation
        self.experiment_size = experiment_size
        self.min_QRAM = min_QRAM
        self.max_QRAM = max_QRAM
        self.size_test_set = len(self.y_test)

        self.acc_8 = []
        self.acc_16 = []
        self.acc_32 = []
        self.acc_64 = []
        self.acc_128 = []


    def experiments_knn(self,k:int = 1,shots:int =10000):
        """
        Execute the Qknn for a specific k 
        Args:
        value: is part of the class 
        k: integer value for the k nearlest neighbor
        shots: integer vlaue for the number of shots i nthe Quantum Circuit

    Returns:
        self.acc8: integer list of accuracies from a QRAM of size 8
        self.acc16: integer list of accuracies from a QRAM of size 16
        self.acc32: integer list of accuracies from a QRAM of size 32
        self.acc64: integer list of accuracies from a QRAM of size 64
        self.acc128: integer list of accuracies from a QRAM of size 128
    """
        for _ in range(self.experiment_size):
            for size in range(self.min_QRAM,self.max_QRAM):
                y_pred = []

                for i in range(self.size_test_set):
                    qc = qknn(self.x_test[i], self.x_train,size,self.features,self.max_trials,self.rotation)
                    result = AerSimulator().run(transpile(qc,basis_gates=["cx","rz","x","sx"],optimization_level=3), shots=shots).result()
                    counts = result.get_counts(qc)
                    counts = {a: v for a, v in sorted(counts.items(), key=lambda item: item[1],reverse=True)}
                    
                    values = list(counts.keys())
                    index = 0 
                    neighbors = {}
                    while index < k:
                        if values[index][0] == "1":
                            k_class = self.y_train[int(values[index][2:],2)]
                            if k_class in neighbors:
                                neighbors[k_class] = neighbors[k_class]+1
                            else:
                                neighbors[k_class] = 1
                            index += 1
                        else:
                            del values[index]
                                
                    knn = {a: v for a, v in sorted(neighbors.items(), key=lambda item: item[1],reverse=True)}
                    y_pred.append(next(iter(knn)))

                
                acc_knn = accuracy_score(self.y_test,y_pred)*100
                if size == 3:
                    self.acc_8.append(acc_knn)
        
                elif size == 4:
                    self.acc_16.append(acc_knn)
        
                elif size == 5:
                    self.acc_32.append(acc_knn)
        
                elif size == 6:
                    self.acc_64.append(acc_knn)
                    
                elif size == 7:
                    self.acc_128.append(acc_knn)
        
    
        return self.acc_8,self.acc_16,self.acc_32,self.acc_64,self.acc_128

    
    def draw_qknn(self,index:int=0,size:int=3):
        """
    Print the qknn quantum circuit
    Args:
        self: is part of the class
        int: the index istance from the test set
        size: integer value abotu the size of the QRAM 

    Returns:
        qknn: the quantum circuit for a particular situation
        """
        return qknn(self.x_test[index], self.x_train,size,self.features,self.max_trials,self.rotation)

    
    def mae_acc(self,acc:list):
        """
    Obtain the MAE  from a lsit of accuracy values
    Args:
    
        self: is part of the class
        acc: a list of float values about the accuracy  

    Returns:
        mean: float value that is the MAE accuracy
        error: float value that is the error from the MAE accuracy
        """
        mean =  np.mean(np.asarray(acc))
        n = len(acc)
        summ = 0
        for i in range(n):
            summ += abs(mean - acc[i])
        return mean,summ/n
    
    def print_results(self):
        """
    Obtain the MAE values for different QRAM sizes: 8,16,32,64,128
    Args:
        self: is part of the class 

    Returns:
        print: the output from the MAE in different QRAM sizes: 8,16,32,64,128
        """
        print("The MAE value of each size is ")
        for i in range(self.min_QRAM,self.max_QRAM):
            mean,error = 0,0
            if i == 3:
                mean,error  = self.mae_acc(self.acc_8)
            elif i == 4:
                mean,error  = self.mae_acc(self.acc_16)
            elif i == 5:
                mean,error  = self.mae_acc(self.acc_32)
            elif i == 6:
                mean,error  = self.mae_acc(self.acc_64)
            elif i == 7:
                mean,error  = self.mae_acc(self.acc_128)
                
            print(f'MAE of  QRAM of size {int(2**i)} cells of memory with {mean:.2f} +/- {error:.2f}.')

>>>>>>> upstream/main
