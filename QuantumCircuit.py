#%%
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm
import plotly.express as px

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator, Aer

from QuantumCNN import QuantumConv2d

def bin_list(num, length):
    """
    Convert a number to a binary list of a given length.

    Args:
        num (int): The number to convert.
        length (int): The length of the binary list.

    Returns:
        list: Binary representation of the number.
    """
    formating = '{0:0' + str(length) + 'b}'
    binary = f'{formating}'.format(num)
    return [int(x) for x in binary]

def build_and_execute_circuit(in_rot, weight_rot):
    # Define the Quantum Circuit
    n_qubits = 4
    qc = QuantumCircuit(n_qubits)
    states = [bin_list(x, n_qubits) for x in range(2**n_qubits)]
    states = np.array(states)

    for i in range(n_qubits):
        qc.h(i)

    for i in range(n_qubits):
        angle = Parameter('enc_0_'+str(i))
        angle = in_rot[i]
        qc.rz(angle,i)

    for i in range(n_qubits-1):
        for j in range(i+1, n_qubits):
            angle = Parameter('enc_1_'+str(i)+'_'+str(j))
            angle = in_rot[i] * in_rot[j]
            qc.rzz(angle, i, j)

    for i in range(n_qubits):
        angle = Parameter('conv_'+str(i))
        angle = weight_rot[i]
        qc.rx(angle, i)

    for i in range(n_qubits):
        qc.cx(i, (i+1) % n_qubits)

    op = Operator(qc)

    # plot the circuit
    qc.draw('mpl')


    test = op.data
    state_probs = np.abs(test)[:,0]**2 / np.sum(np.abs(test)[:,0]**2)
    qubit_probs = state_probs @ states
    
    return qubit_probs

#%%
if __name__ == '__main__':
    qcon = QuantumConv2d(2, 2, 8)
    x = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device='cuda')
    qcon.weight = nn.Parameter(torch.tensor([[0.5, 0.6], [0.7, 0.8]], device='cuda'))
    qcon.sub_forward(x)
    
    in_rot = [0.1, 0.2, 0.3, 0.4]
    weight_rot = [0.5, 0.6, 0.7, 0.8]
    build_and_execute_circuit(in_rot, weight_rot)
