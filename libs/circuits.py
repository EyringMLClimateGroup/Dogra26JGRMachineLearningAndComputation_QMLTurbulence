import pennylane as qml
import numpy as np

class PQCircuit:
    def __init__(self, n_enc, n_dec, n_wires, var_angles = 'XYZ', encoding_angles='X', choose_freq = True):
        self.n_enc = n_enc #number of encoding layers
        self.n_dec = n_dec #number of decoding layers
        self.n_wires = n_wires #number of qubits
        self.var_angles = var_angles #types of trainable rotation angles (X,Y,Z)
        self.n_var_angles = len(var_angles)*(n_enc + n_dec)*n_wires #number of trainable rotation angles
        self.encoding_angles = encoding_angles #types of encoding angles (X,Y,Z)
        self.n_encoding_angles = len(np.unique(list(encoding_angles)))*n_enc*n_wires #number of encoding angles
        self.choose_freq = choose_freq #flag whether the frequencies are trainable via a trainable weight in the data upload
        self.circuit_name = 'circuit_'+encoding_angles+'_'+var_angles+'_freq'*choose_freq #unique name for each type of circuit architecture
    
    def __call__(self, data, pars, wires):
        return globals()[self.circuit_name](data, pars, self.n_enc, self.n_dec, wires)

# --- Quantum Circuit Layouts ---

def circuit_X_XYZ_freq(data, pars, n_enc, n_dec, wires):
    n_wires = len(wires)
    expected = n_wires*(4*n_enc + 3*n_dec)
    assert len(pars) == expected, f"pars must have length {expected}, has length {len(pars)}"
    idx = n_enc*n_wires
    idy = 0
    for layer in range(n_enc + n_dec):
        ### data uploads
        if layer < n_enc:
            qml.AngleEmbedding(data*pars[idy:idy + n_wires], wires=wires, rotation='X')
            idy += n_wires
            
        ### variational gates
        angles = pars[idx:idx + 3 * n_wires]
        for q in range(n_wires):
            qml.RX(angles[q*3], wires=q)
            qml.RY(angles[q*3+1], wires=q)
            qml.RZ(angles[q*3+2], wires=q)
        idx += 3 * n_wires

        ### entangling layer
        for i in range(n_wires):
            qml.CNOT(wires=[wires[i], wires[(i+1)%n_wires]])

def circuit_XX_XYZ_freq(data, pars, n_enc, n_dec, wires):
    n_wires = len(wires)
    n_features = n_wires//2
    expected = n_wires*(4*n_enc + 3*n_dec)
    assert len(pars) == expected, f"pars must have length {expected}, has length {len(pars)}"
    
    idx = n_enc*n_wires
    idy = 0
    for layer in range(n_enc + n_dec):
        ### data uploads
        if layer < n_enc:
            qml.AngleEmbedding(data*pars[idy:idy + n_features], wires=wires[:n_features], rotation='X')
            qml.AngleEmbedding(data*pars[idy + n_features:idy + n_wires], wires=wires[n_features:], rotation='X')
            idy += n_wires
            
        ### variational gates
        angles = pars[idx:idx + 3 * n_wires]
        #print(pars.shape, angles.shape, idx)
        for q in range(n_wires):
            qml.RX(angles[q*3], wires=q)
            qml.RY(angles[q*3+1], wires=q)
            qml.RZ(angles[q*3+2], wires=q)
        idx += 3 * n_wires

        ### entangling layer
        for i in range(n_wires):
            qml.CNOT(wires=[wires[i], wires[(i+1)%n_wires]])

def circuit_XY_XYZ_freq(data, pars, n_enc, n_dec, wires, n_features = 6):
    n_wires = len(wires)
    n_fw = min(n_wires, n_features)
    idx = 2*n_enc*n_fw
    idy = 0
    for layer in range(n_enc + n_dec):
        ### data uploads
        if layer < n_enc:
            qml.AngleEmbedding(data[:,:n_fw]*pars[idy:idy + n_fw], wires=wires[:n_fw], rotation='X')
            qml.AngleEmbedding(data[:,-n_fw:]*pars[idy + n_fw:idy + 2*n_fw], wires=wires[-n_fw:], rotation='Y')
            idy += 2*n_fw
            
        ### variational gates
        angles = pars[idx:idx + 3 * n_wires]
        for q in range(n_wires):
            qml.RX(angles[q*3], wires=q)
            qml.RY(angles[q*3+1], wires=q)
            qml.RZ(angles[q*3+2], wires=q)
        idx += 3*n_wires

        ### entangling layer
        for i in range(n_wires):
            qml.CNOT(wires=[wires[i], wires[(i+1)%n_wires]])

def circuit_XYZ_XYZ_freq(data, pars, n_enc, n_dec, wires, n_features = 6):
    n_wires = len(wires)
    idx = 3*n_enc*n_wires
    idy = 0
    n_fw = max(n_wires, n_features)
    def id_x(start):
        return (np.arange(start, start + n_wires) % n_features)
    for layer in range(n_enc + n_dec):
        ### data uploads
        if layer < n_enc:
            qml.AngleEmbedding(data[:,id_x(0)]*pars[idy:idy + n_wires], wires=wires, rotation='X')
            qml.AngleEmbedding(data[:,id_x(n_wires)]*pars[idy + n_wires:idy + 2*n_wires], wires=wires, rotation='Y')
            qml.AngleEmbedding(data[:,id_x(2*n_wires)]*pars[idy + 2*n_wires:idy + 3*n_wires], wires=wires, rotation='Z')
            idy += 3*n_wires
            
        ### variational gates
        angles = pars[idx:idx + 3 * n_wires]
        for q in range(n_wires):
            qml.RX(angles[q*3], wires=q)
            qml.RY(angles[q*3+1], wires=q)
            qml.RZ(angles[q*3+2], wires=q)
        idx += 3 * n_wires

        ### entangling layer
        for i in range(n_wires):
            qml.CNOT(wires=[wires[i], wires[(i+1)%n_wires]])

#--------------circuits with fixed uploading factor 1 (not used in the publication) -----------------

def circuit_X_XYZ(data, pars, n_enc, n_dec, wires):
    n_wires = len(wires)
    idx = 0
    for layer in range(n_enc + n_dec):
        ### data uploads
        if layer < n_enc:
            qml.AngleEmbedding(data, wires=wires, rotation='X')
            
        ### variational gates
        angles = pars[idx:idx + 3 * n_wires]
        for q in range(n_wires):
            qml.RX(angles[q*3], wires=q)
            qml.RY(angles[q*3+1], wires=q)
            qml.RZ(angles[q*3+2], wires=q)
        idx += 3 * n_wires

        ### strongly entangling layer
        for i in range(n_wires):
            qml.CNOT(wires=[wires[i], wires[(i+1)%n_wires]])


def circuit_X_YZ(data, pars, n_enc, n_dec, wires):
    n_wires = len(wires)
    idx = 0
    for layer in range(n_enc + n_dec):
        ### data uploads
        if layer < n_enc:
            qml.AngleEmbedding(data, wires=wires, rotation='X')
            
        ### variational gates
        angles = pars[idx:idx + 2*n_wires]
        for q in range(n_wires):
            qml.RY(angles[q*2], wires=q)
            qml.RZ(angles[q*2 + 1], wires=q)
        idx += 2*n_wires

        ### strongly entangling layer
        for i in range(n_wires):
            qml.CNOT(wires=[wires[i], wires[(i+1)%n_wires]])

def circuit_X_XY(data, pars, n_enc, n_dec, wires):
    n_wires = len(wires)
    idx = 0
    for layer in range(n_enc + n_dec):
        ### data uploads
        if layer < n_enc:
            qml.AngleEmbedding(data, wires=wires, rotation='X')
            
        ### variational gates
        angles = pars[idx:idx + 2*n_wires]
        for q in range(n_wires):
            qml.RX(angles[q*2], wires=q)
            qml.RY(angles[q*2 + 1], wires=q)
        idx += 2*n_wires

        ### strongly entangling layer
        for i in range(n_wires):
            qml.CNOT(wires=[wires[i], wires[(i+1)%n_wires]])

def circuit_X_Y(data, pars, n_enc, n_dec, wires):
    n_wires = len(wires)
    idx = 0
    for layer in range(n_enc + n_dec):
        ### data uploads
        if layer < n_enc:
            qml.AngleEmbedding(data, wires=wires, rotation='X')
            
        ### variational gates
        angles = pars[idx:idx + n_wires]
        for q in range(n_wires):
            qml.RY(angles[q], wires=q)
        idx += n_wires

        ### strongly entangling layer
        for i in range(n_wires):
            qml.CNOT(wires=[wires[i], wires[(i+1)%n_wires]])

def circuit_XY_XYZ(data, pars, n_enc, n_dec, wires):
    ## data needs twice as many features as n_wires!
    n_wires = len(wires)
    idx = 0
    for layer in range(n_enc + n_dec):
        ### data uploads
        if layer < n_enc:
            qml.AngleEmbedding(data[:n_wires], wires=wires, rotation='X')
            qml.AngleEmbedding(data[n_wires:], wires=wires, rotation='Y')
            
        ### variational gates
        angles = pars[idx:idx + 3 * n_wires]
        for q in range(n_wires):
            qml.RX(angles[q*3], wires=q)
            qml.RY(angles[q*3+1], wires=q)
            qml.RZ(angles[q*3+2], wires=q)
        idx += 3 * n_wires

        ### strongly entangling layer
        for i in range(n_wires):
            qml.CNOT(wires=[wires[i], wires[(i+1)%n_wires]])

def circuit_XY_XYZ(data, pars, n_enc, n_dec, wires):
    ## data needs twice as many features as n_wires!
    n_wires = len(wires)
    n_features = data.shape[1]
    
    if n_wires < n_features:
        idx = 0
        for layer in range(n_enc + n_dec):
            ### data uploads
            if layer < n_enc:
                qml.AngleEmbedding(data[:,:n_wires], wires=wires, rotation='X')
                qml.AngleEmbedding(data[:,n_wires:], wires=wires, rotation='Y')
                
            ### variational gates
            angles = pars[idx:idx + 3 * n_wires]
            for q in range(n_wires):
                qml.RX(angles[q*3], wires=q)
                qml.RY(angles[q*3+1], wires=q)
                qml.RZ(angles[q*3+2], wires=q)
            idx += 3 * n_wires
    
            ### strongly entangling layer
            for i in range(n_wires):
                qml.CNOT(wires=[wires[i], wires[(i+1)%n_wires]])
                
    elif n_features < n_wires:
        idx = 0
        for layer in range(n_enc + n_dec):
            ### data uploads
            if layer < n_enc:
                qml.AngleEmbedding(data, wires=wires[:n_features], rotation='X')
                qml.AngleEmbedding(data, wires=wires[n_wires - n_features:], rotation='Y')
                
            ### variational gates
            angles = pars[idx:idx + 3 * n_wires]
            for q in range(n_wires):
                qml.RX(angles[q*3], wires=q)
                qml.RY(angles[q*3+1], wires=q)
                qml.RZ(angles[q*3+2], wires=q)
            idx += 3 * n_wires
    
            ### strongly entangling layer
            for i in range(n_wires):
                qml.CNOT(wires=[wires[i], wires[(i+1)%n_wires]])
