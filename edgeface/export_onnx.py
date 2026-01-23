import torch
import torch.nn as nn
from backbones import get_model
import os

# CONFIGURATION
MODEL_NAME = 'edgeface_xxs'
CHECKPOINT_PATH = 'edgeface_xxs.pt'
OUTPUT_PATH = 'edgeface_xxs.onnx'

def export():
    print(f"Current working directory: {os.getcwd()}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found!")
        return

    print(f"Loading {MODEL_NAME}...")
    try:
        model = get_model(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model definition: {e}")
        return
    
    print("Loading weights...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    model.eval()

    # Create Dummy Input (112x112)
    dummy_input = torch.randn(1, 3, 112, 112)

    print(f"Exporting to {OUTPUT_PATH}...")
    try:
        # Force legacy TorchScript exporter to avoid external data files
        torch.onnx.export(
            model,
            dummy_input,
            OUTPUT_PATH,
            input_names=['input'],
            output_names=['embedding'],
            dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
            opset_version=14,
            export_params=True,
            do_constant_folding=True,
            dynamo=False  # Force legacy TorchScript exporter
        )
        print("Export successful!")
        
        # Verify file was created
        if os.path.exists(OUTPUT_PATH):
            size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
            print(f"Created {OUTPUT_PATH}: {size_mb:.2f} MB")
            # Check if external data file was created
            data_file = OUTPUT_PATH + ".data"
            if os.path.exists(data_file):
                print(f"WARNING: External data file {data_file} was created!")
                print("The model may not load correctly in the browser worker.")
            else:
                print("No external data file - model is self-contained!")
    except Exception as e:
        print(f"Error during export: {e}")

if __name__ == '__main__':
    export()
