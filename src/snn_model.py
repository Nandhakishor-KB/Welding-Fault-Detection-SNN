import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer

class WeldingSNN(nn.Module):
    def __init__(self, num_inputs=13, num_classes=5, tau=2.0):
        """
        Spiking Neural Network for Welding Fault Detection.
        
        Args:
            num_inputs (int): Number of MFCC features (Default: 13)
            num_classes (int): Number of fault categories (Default: 5)
            tau (float): Membrane time constant for LIF neurons
        """
        super().__init__()
        
        # Layer 1: Input encoding to hidden spiking layer
        self.layer1 = nn.Sequential(
            layer.Linear(num_inputs, 64),
            neuron.LIFNode(tau=tau, detach_reset=True)
        )
        
        # Layer 2: Hidden to Output
        self.layer2 = nn.Sequential(
            layer.Linear(64, num_classes),
            neuron.LIFNode(tau=tau, detach_reset=True)
        )

    def forward(self, x):
        # x shape: [Batch, Time_Steps, Features]
        # We process the time dimension using functional.multi_step_forward
        return functional.multi_step_forward(x, self.layer1, self.layer2)

if __name__ == "__main__":
    # Test the model structure
    model = WeldingSNN()
    print("SNN Architecture Created:")
    print(model)
