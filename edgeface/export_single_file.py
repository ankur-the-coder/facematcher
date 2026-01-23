import torch
import onnx
from backbones import get_model
import os

# CONFIGURATION
MODEL_NAME = 'edgeface_xxs'
CHECKPOINT_PATH = 'edgeface_xxs.pt'
OUTPUT_PATH = 'edgeface_xxs_single.onnx'

def export():
    print(f"Current working directory: {os.getcwd()}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found!")
        return

    print(f"Loading {MODEL_NAME}...")
    model = get_model(MODEL_NAME)
    
    print("Loading weights...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # Create Dummy Input (112x112)
    dummy_input = torch.randn(1, 3, 112, 112)

    print(f"Exporting to {OUTPUT_PATH}...")
    
    # Use TorchScript to trace the model first
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)
    
    # Export with legacy exporter - smaller threshold
    torch.onnx.export(
        traced,
        dummy_input,
        OUTPUT_PATH,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
        opset_version=14,
        export_params=True,
        do_constant_folding=True,
        dynamo=False
    )
    
    # Check if external data was created
    data_file = OUTPUT_PATH + ".data"
    if os.path.exists(data_file):
        print(f"External data file was created. Attempting to convert to single file...")
        
        # Load the ONNX model
        model_onnx = onnx.load(OUTPUT_PATH)
        
        # Save with external data embedded
        onnx.save_model(
            model_onnx,
            OUTPUT_PATH.replace('_single', '_embedded'),
            save_as_external_data=False,
            all_tensors_to_one_file=False,
            location=None,
            size_threshold=0,  # Embed everything
            convert_attribute=True
        )
        
        print(f"Created {OUTPUT_PATH.replace('_single', '_embedded')}")
        
        # Clean up
        final_path = 'edgeface_xxs_final.onnx'
        os.rename(OUTPUT_PATH.replace('_single', '_embedded'), final_path)
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)
        if os.path.exists(data_file):
            os.remove(data_file)
        
        print(f"Final model: {final_path}")
        print(f"Size: {os.path.getsize(final_path) / (1024*1024):.2f} MB")
    else:
        print(f"Success! No external data file.")
        final_path = 'edgeface_xxs_final.onnx'
        os.rename(OUTPUT_PATH, final_path)
        print(f"Final model: {final_path}")
        print(f"Size: {os.path.getsize(final_path) / (1024*1024):.2f} MB")

if __name__ == '__main__':
    export()
