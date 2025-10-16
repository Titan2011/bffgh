import torch
from RandLANet import Network

# 1. Create a dummy config class that mimics your cfg object
#    Fill in the values based on your actual configuration.
class DummyConfig:
    def __init__(self):
        self.num_classes = 8      # From your logs
        self.num_features = 6     # From your logs (xyz + rgb)
        self.num_layers = 5       # A typical value for RandLANet, adjust if needed
        self.d_out = [32, 64, 128, 256, 256] # Example, adjust to your config
        self.sub_sampling_ratio = [1, 3, 4, 4, 2] # Example, adjust
        self.k_n = 16             # Example, adjust

def check_model_parameters(model):
    """
    Analyzes the model's parameters and prints a summary.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("           MODEL PARAMETER ANALYSIS")
    print("="*60)
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Non-Trainable (Frozen): {total_params - trainable_params:,}")
    print("="*60 + "\n")

    if trainable_params < 100000: # A real network should have many more
        print("!! DIAGNOSIS: The model is not constructed correctly. !!")
        print("The number of trainable parameters is far too low.")
        print("This confirms that most of the network is 'frozen' or")
        print("disconnected from the computation graph.\n")
    else:
        print("DIAGNOSIS: The model parameter count looks healthy.")
        print("The issue may lie elsewhere, but the model structure is likely correct.\n")

    # Optional: uncomment this to see the status of EVERY layer.
    # print("--- Detailed Layer Status ---")
    # for name, param in model.named_parameters():
    #     print(f"{name:<60} | Trainable: {param.requires_grad} | Size: {param.numel():,}")
    # print("-" * 60)


if __name__ == '__main__':
    print("Initializing model for debugging...")
    
    # 2. Instantiate the dummy config and the Network
    config = DummyConfig()
    model = Network(config)
    
    print("Model initialized. Now checking parameters...")
    
    # 3. Run the analysis
    check_model_parameters(model)
