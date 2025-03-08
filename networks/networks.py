import torch.nn as nn
import torch.nn.functional as F

from . import utils


class LeNet(utils.ReparamModule):
    supported_dims = {28, 32}

    def __init__(self, state):
        if state.dropout:
            raise ValueError("LeNet doesn't support dropout")
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(state.nc, 6, 5, padding=2 if state.input_size == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 if state.num_classes <= 2 else state.num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out

# QNN
class QNN(utils.ReparamModule):
    supported_dims = {28, 32}

    def __init__(self, state):
        super(QNN, self).__init__()
        self.num_qubits = 6
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.n_layers = 3
        self.n_qnns = 1
        self.diag = nn.Parameter(torch.randn(self.num_qubits, 2), requires_grad=True)
        self.off_diag = nn.Parameter(torch.randn(self.num_qubits), requires_grad=True)

        def herm_matrix(i):
            """Builds guaranteed-Hermitian 2x2 matrix"""
            mat = torch.stack([
                torch.stack([self.diag[i,0], self.off_diag[i]]),
                torch.stack([self.off_diag[i], self.diag[i,1]])
            ])
            return 0.5 * (mat + mat.T)

        @qml.qnode(self.dev, interface="torch")
        def qnn_pqc(inputs, weights):
            qml.templates.AmplitudeEmbedding(inputs, wires=range(self.num_qubits), normalize=True)
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return [qml.expval(qml.Hermitian(herm_matrix(i), wires=[i])) for i in range(self.num_qubits)]
            
        weight_shapes = {"weights": (self.n_layers, self.num_qubits, 3)}

        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = 1 - self.alpha
        self.conv1 = nn.Conv2d(state.nc, 6, 5, padding=2 if state.input_size == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.qnns = qml.qnn.TorchLayer(qnn_pqc, weight_shapes, init_method=torch.nn.init.xavier_normal_)
        self.fc3 = nn.Linear(self.num_qubits, 1 if state.num_classes <= 2 else state.num_classes)
        self.fc_residual = nn.Linear(64, 1 if state.num_classes <= 2 else state.num_classes)
        self.fc_final = nn.Linear(1 if state.num_classes <= 2 else state.num_classes, 1 if state.num_classes <= 2 else state.num_classes)
    
    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out_for_residue = self.fc2(out)
        out = self.qnns(out_for_residue)
        out = self.alpha * self.fc3(out) + self.beta * F.relu(self.fc_residual(out_for_residue), inplace=True)
        out = self.fc_final(out)
        return out

class AlexCifarNet(utils.ReparamModule):
    supported_dims = {32}

    def __init__(self, state):
        super(AlexCifarNet, self).__init__()
        assert state.nc == 3
        self.features = nn.Sequential(
            nn.Conv2d(state.nc, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, state.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = self.classifier(x)
        return x


# ImageNet
class AlexNet(utils.ReparamModule):
    supported_dims = {224}

    class Idt(nn.Module):
        def forward(self, x):
            return x

    def __init__(self, state):
        super(AlexNet, self).__init__()
        self.use_dropout = state.dropout
        assert state.nc == 3 or state.nc == 1, "AlexNet only supports nc = 1 or 3"
        self.features = nn.Sequential(
            nn.Conv2d(state.nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if state.dropout:
            filler = nn.Dropout
        else:
            filler = AlexNet.Idt
        self.classifier = nn.Sequential(
            filler(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            filler(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1 if state.num_classes <= 2 else state.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
