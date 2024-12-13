#%%
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm

# Define quantum gates
gates = {
    'H': torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / torch.sqrt(torch.tensor([2.0], dtype=torch.cfloat)),
    'X': torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat),
    'I': torch.eye(2, dtype=torch.cfloat),
}

class Hadamard(nn.Module):
    """
    A Hadamard gate.
    this module is a fixed transformation of the input values.
    """
    def __init__(self, qubits):
        """
        Initialize the Hadamard gate.
        """
        super(Hadamard, self).__init__()
        self.qubits = qubits
        self.register_buffer('hadamard', gates['H'], persistent=True)
        self.register_buffer('unitary', self.get_Hadamard(qubits), persistent=True)

    def get_Hadamard(self, qubits):
        """
        Get the Hadamard gate for given qubits.

        Args:
            qubits (int): Number of qubits.

        Returns:
            torch.Tensor: Hadamard gate matrix.
        """
        unitary = self.hadamard
        for _ in range(1, qubits):
            unitary = torch.kron(unitary, self.hadamard)
        return unitary

    def forward(self, x:torch.Tensor):
        """
        Perform the forward pass of the Hadamard gate.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return x.matmul(self.unitary[None,:,:])


class CNOTRing(nn.Module):
    """
    A CNOT ring gate.
    """
    def __init__(self, qubits):
        """
        Initialize the CNOT ring gate.
        this module is a fixed transformation of the input values.

        Args:
            num_qubits (int): Number of qubits.
        """
        super(CNOTRing, self).__init__()
        self.qubits = qubits
        self.register_buffer('unitary', self.get_CNOT_ring(), persistent=True)

    def get_CNOT(self, control, target, qubits):
        """
        Get the CNOT gate for given control and target qubits.

        Args:
            control (int): Control qubit index.
            target (int): Target qubit index.
            qubits (int): Number of qubits.

        Returns:
            torch.Tensor: CNOT gate matrix.
        """
        swap = True
        if control > target:
            swap = False
            control, target = target, control
        diff = target - control
        if diff > 1:
            scaler = torch.eye(2**(diff-1))
            upper = torch.kron(scaler, gates['I'])
            lower = torch.kron(scaler, gates['X'])
        else:
            upper = gates['I']
            lower = gates['X']
        
        unitary = torch.kron(torch.tensor([[1, 0], [0, 0]]), upper) + torch.kron(torch.tensor([[0, 0], [0, 1]]), lower)

        if swap:
            swap_matrix = gates['H']
            for _ in range(1,diff+1):
                swap_matrix = torch.kron(swap_matrix, gates['H'])
            unitary = swap_matrix @ unitary @ swap_matrix

        if qubits > diff + 1:
            bits_before = int(control - 1)
            bits_after = int(qubits - target)
            unitary = torch.kron(torch.eye(2**bits_after), torch.kron(unitary, torch.eye(2**bits_before)))

        return unitary

    def get_CNOT_ring(self):
        """
        Get the CNOT ring for all qubits.

        Args:
            num_qubits (int): Number of qubits.

        Returns:
            torch.Tensor: CNOT ring matrix.
        """
        unitary = self.get_CNOT(1, 2, self.qubits)
        for i in range(2, self.qubits):
            unitary = self.get_CNOT(i, i+1, self.qubits) @ unitary
        unitary = self.get_CNOT(self.qubits, 1, self.qubits) @ unitary
        return unitary
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the CNOT ring gate.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.unitary[None,:,:].matmul(x)


class HCNOT(nn.Module):
    """
    An HCNOT gate.
    """
    def __init__(self, qubits):
        """
        Initialize the HCNOT gate.
        """
        super(HCNOT, self).__init__()
        self.qubits = qubits
        self.h = Hadamard(qubits)
        self.cnot = CNOTRing(qubits)

    def forward(self, x):
        """
        Perform the forward pass of the HCNOT gate.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.h(self.cnot(x))


class RX(nn.Module):
    """
    An RX gate.
    """
    def __init__(self, qubits):
        """
        Initialize the RX gate.

        Args:
            rotation (torch.Tensor): Rotation angle.
        """
        super(RX, self).__init__()
        self.qubits = qubits
        self.shape = (self.qubits//2, self.qubits//2)
        shift = torch.kron(torch.ones(self.qubits, dtype=torch.cfloat), torch.pi/2 * torch.eye(self.shape[0], dtype=torch.cfloat))
        signer = torch.kron(torch.ones(self.qubits, dtype=torch.cfloat), torch.tensor([[1, -1j], [-1j, 1]], dtype=torch.cfloat))
        self.register_buffer('signer', signer, persistent=True)
        self.register_buffer('shift', shift, persistent=True)
        self.register_buffer('pattern', torch.ones(self.shape, dtype=torch.cfloat), persistent=True)

        self.weight = nn.Parameter(torch.randn(qubits)) # trainable parameter

    def forward(self, x):
        """
        Perform the forward pass of the RX gate.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        scale = torch.kron(self.weight/2, self.pattern)
        w_scale = scale + self.shift
        w_scale = torch.sin(w_scale)
        w_scale = self.signer * w_scale
        
        unitary = w_scale[:,-2:]
        for i in range(1, self.qubits):
            idx = -2 * i
            unitary = torch.kron(unitary, w_scale[:,idx-2:idx])

        return unitary[None,:,:].matmul(x)


class RZRZZ(nn.Module):
    """
    An RZRZZ gate.
    """
    def __init__(self, qubits):
        """
        Initialize the RZRZZ gate.
        This module acts as a higher order encoding of the input values.

        Args:
            qubits (int): Number of qubits.
        """
        super(RZRZZ, self).__init__()
        self.qubits = qubits
        qubit_tuples = torch.tensor([[i, j] for i in range(1, self.qubits+1) for j in range(i+1, self.qubits+1)]).int()
        self.register_buffer('qubit_tuples', qubit_tuples, persistent=True)
        rzz_diags = []
        for i in range(len(qubit_tuples)):
            rzz_diags.append(self.get_RZZ_static(qubit_tuples[i]))
        rzz_diags = torch.stack(rzz_diags)
        self.register_buffer('rzz_diags', rzz_diags, persistent=True)
        self.register_buffer('rzzscaler', torch.eye(2**self.qubits, dtype=torch.cfloat), persistent=True)
        self.register_buffer('rzfactor', torch.tensor([2 * 1j], dtype=torch.cfloat), persistent=True)
        upper_triag = torch.triu(torch.arange(self.qubits**2).reshape(self.qubits, self.qubits), diagonal=1)
        upper_triag = upper_triag[upper_triag != 0]
        self.register_buffer('upper_triag', upper_triag, persistent=True)
        self.register_buffer('idx', torch.arange(len(qubit_tuples))[:,None], persistent=True)
        self.register_buffer('rz_static', self.get_static_RZ(), persistent=True)

    def get_static_RZ(self):
        """
        Get the static RZ rotation matrix for all qubits.

        Args:
            qubits (int): Number of qubits.

        Returns:
            torch.Tensor: Static RZ rotation matrix.
        """
        binary_array = [self.bin_list(x, self.qubits) for x in range(2**self.qubits)]
        binary_array = torch.flip(torch.tensor(binary_array), dims=(-1,-2))
        sign_matrix = -torch.ones((2**self.qubits, self.qubits)) + 2 * binary_array
        return sign_matrix

    def get_RZZ_static(self, qubits):
        """
        Get the RZZ gate for given qubits and rotation.

        Args:
            qubits (int): Number of qubits.
            rotation (torch.Tensor): Rotation angle.

        Returns:
            torch.Tensor: RZZ gate matrix.
        """
        control = torch.min(qubits)
        target = torch.max(qubits)
        diff = target - control
        upper_diff = 4 - target
        b1 = torch.tensor([1j*1])
        b2 = torch.tensor([-1j*1])
        operator_core = torch.concat([b2, b1, b1, b2])
        
        operator = torch.kron(operator_core, torch.ones(2**(control-1)))

        if diff > 1:
            operator_upper = operator[:len(operator)//2]
            operator_lower = operator[len(operator)//2:]
            scaler = torch.ones(2**(diff-1))
            upper = torch.kron(scaler, operator_upper)
            lower = torch.kron(scaler, operator_lower)
            operator = torch.kron(torch.tensor([1, 0]), upper) + torch.kron(torch.tensor([0, 1]), lower)
        
        if upper_diff > 0:
            operator = torch.kron(torch.ones(2**upper_diff), operator)

        return operator

    def bin_list(self, num, length):
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
    
    def get_RZZ(self, idx, rotation):
        """
        Get the RZZ gate for given qubits and rotation.

        Args:
            qubits (int): Number of qubits.
            rotation (torch.Tensor): Rotation angle.

        Returns:
            torch.Tensor: RZZ gate matrix.
        """
        tmp_diag = self.rzz_diags[idx]
        tmp_diag = torch.exp(tmp_diag*rotation/2)
        unitary = tmp_diag * self.rzzscaler

        return unitary

    def forward(self, x:torch.Tensor):
        """
        Perform the forward pass of the RZRZZ gate.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        rot_mul = x[:,:,None].matmul(x[:,None,:])
        rot_mul = rot_mul.flatten(-2)[:,self.upper_triag]
        batch_dim = x.shape[0]
        tmp_diag = self.rzz_diags[self.idx.T.repeat(batch_dim,1)]
        tmp_diag = torch.exp(rot_mul[:,:,None]/2 * tmp_diag)
        ops = tmp_diag[:,:,:,None] * self.rzzscaler[None,None,:,:]
        # batch, 6, 16, 16
        unitary_rzz = ops[:,-1,:,:]
        for i in range(0, ops.shape[1]-1):
            unitary_rzz = unitary_rzz.matmul(ops[:,ops.shape[1]-2-i,:])

        rot = x.div(self.rzfactor)
        unitary_rz = torch.sum(self.rz_static[None,:,:] * rot[:,None,:], dim=(-1,))
        unitary_rz = torch.exp(unitary_rz)
        unitary_rz = unitary_rz[:,:,None] * self.rzzscaler[None,:,:]
        
        return unitary_rzz.matmul(unitary_rz)


class QuantumConv2d(nn.Module):
    """
    A quantum convolutional layer for 2D inputs.
    """
    def __init__(self, kernel_size, stride, size, device=None):
        """
        Initialize the QuantumConv2d layer.

        Args:
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            size (int): Size of the input.
        """
        super(QuantumConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.qubits = kernel_size**2
        self.size = size

        self.hcnot = HCNOT(self.qubits)
        self.rx = RX(self.qubits)
        self.rzrzz = RZRZZ(self.qubits)

        # Permutation for sub-parts of the input
        sub_part_permutation = ((torch.arange(0,(size**2)/kernel_size) % size)* (size/kernel_size) 
                                     + torch.arange(0,(size**2)/kernel_size) // size).int()
        divisor = self.size // self.kernel_size
        permute_back = (torch.arange(0, self.size**2//self.kernel_size) % divisor * self.size
                             + torch.arange(0, self.size**2//self.kernel_size) // divisor).int()
        
        self.unfolder = nn.Unfold((kernel_size,kernel_size), stride=stride)

        self.register_buffer('sub_part_permutation', sub_part_permutation, persistent=True)
        self.register_buffer('permute_back', permute_back, persistent=True)
        self.register_buffer('states', self.get_static_state_list(self.qubits), persistent=True)
    
    def bin_list(self, num, length):
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
    
    def get_static_state_list(self, qubits):
        """
        Get the static state list for all qubits.

        Args:
            qubits (int): Number of qubits.

        Returns:
            torch.Tensor: Static state list.
        """
        states = [self.bin_list(x, qubits) for x in range(2**qubits)]
        states = torch.tensor(states, dtype=torch.float32)
        return states
    
    def sub_forward(self, x: torch.Tensor):
        """
        NOTE This function simulates the quantum circuit for a sub-part of the input.
        Perform the forward pass for a sub-part of the input.

        Args:
            x (torch.Tensor): Input tensor. shape:(batch*(pixelanzahl/kernel_size), kernel_size, kernel_size)

        Returns:
            torch.Tensor: Output tensor. shape:(batch*(pixelanzahl/kernel_size), kernel_size, kernel_size)
        """
        # Define the sequence of operations in the quantum circuit
        x = self.rzrzz(x) # encode the input
        x = self.rx(x) # apply weights
        x = self.hcnot(x) # transform the input

        # Calculate the state probabilities e.g. measure the output
        state_probs = (torch.abs(x[:, :, 0])**2)
        state_probs = state_probs.matmul(self.states)

        return state_probs
    
    def forward(self, x:torch.Tensor):
        """
        Perform the forward pass of the QuantumConv2d layer.

        Folding info:
         __ __ __ __ 
        |---->|---->|
        |---->|---->|
        ------------
        |---->|---->|
        |---->|---->|
        ------------

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_dim = x.shape[0]
        img_dim = x.shape[-1]
        # Unfold the input tensor
        x = self.unfolder(x)
        x = x.permute(0,2,1).flatten(end_dim=-2)
        # NOTE Quantum Circuit
        x = self.sub_forward(x)
        x = x.reshape(batch_dim, img_dim//self.kernel_size, img_dim//self.kernel_size, self.qubits)
        x = x.permute(0,3,1,2)
        return x


class ClassicalConvNet(nn.Module):
    """
    A quantum convolutional neural network.
    """
    def __init__(self):
        """
        Initialize the QuantumConvNet.
        """
        super(ClassicalConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 2, 2)
        self.fc1 = nn.Linear(28**2, 10)

    def forward(self, x):  
        """
        Perform the forward pass of the QuantumConvNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = torch.relu(self.conv1(x))
        # 4 * 4 * 4 = 64
        x = x.flatten(1)
        x = torch.softmax(self.fc1(x), dim=-1)
        return x


class QuantumConvNet(nn.Module):
    """
    A quantum convolutional neural network.
    """
    def __init__(self, dev=None):
        """
        Initialize the QuantumConvNet.
        """
        super(QuantumConvNet, self).__init__()
        self.qconv = QuantumConv2d(2, 2, 28, device=dev)
        self.fc1 = nn.Linear(28**2, 10)
    
    def forward(self, x):  
        """
        Perform the forward pass of the QuantumConvNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.qconv(x)
        # 4 * 4 * 4 = 64
        x = x.flatten(1)
        x = torch.softmax(self.fc1(x), dim=-1)
        return x
    
#%%

if __name__ == '__main__':
    #%% Data transformation and loading
    import plotly.express as px
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.MNIST('./', download=True, train=True, transform=transform)
    # cut dataset to 1000 samples
    trainset.data = trainset.data[:10000]
    valset = datasets.MNIST('./', download=True, train=False, transform=transform)
    # cut dataset to 100 samples
    valset.data = valset.data[:1000]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=10, shuffle=True)

    # Initialize the quantum convolutional neural network
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #dev = 'meta'    
    qnet = ClassicalConvNet()
    qnet = qnet.to(device=dev)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(qnet.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(qnet.parameters(), lr=0.001)
    #print(qnet.qconv.rx.weight)
    test = qnet.fc1.weight

    from torchview import draw_graph
    model_graph = draw_graph(qnet, input_size=(10,1,28,28), device=dev)
    model_graph.visual_graph

    # Training loop
    for epoch in tqdm(range(10)):
        running_loss = []
        for i, (X_batch, y_batch) in enumerate(trainloader):
            
            optimizer.zero_grad()

            outputs = qnet(X_batch.to(dev))
            loss = criterion(outputs, y_batch.to(dev))
            loss.retain_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

        print(f'Epoch: {epoch}, Loss: {np.mean(running_loss)}')
        #print(qnet.qconv.rx.weight)
        #print(qnet.qconv.rx.weight.grad)
        running_loss = []

        # Validation
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in valloader:
                outputs = qnet(X_batch.to(dev))
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch.to(dev)).sum().item()

        print(f'Accuracy: {100 * correct / total}')

    print('Finished Training')

    #%% computational graph of quantum convolutional layer
