# Model Conversion Guide

The application requires **ONNX** models to run in the browser. You currently have a PyTorch (`.pt`) file.

## 1. Convert EdgeFace to ONNX

Since the model definition is required to load the weights, you need to use the official repository to perform the conversion.

### Prerequisites
- Python installed
- `torch`, `onnx` installed (`pip install torch onnx`)
- The `edgeface` repository

### Steps
1.  Clone the EdgeFace repository:
    ```bash
    git clone https://github.com/otroshi/edgeface.git
    cd edgeface
    ```
2.  Copy your `edgeface_xxs.pt` file into this directory.
3.  Create a file named `export_onnx.py` in the `edgeface` directory with the following content:

    ```python
    import torch
    import torch.nn as nn
    from backbones import get_model  # This comes from the repo

    # CONFIGURATION
    MODEL_NAME = 'edgeface_xxs'
    CHECKPOINT_PATH = 'edgeface_xxs.pt'
    OUTPUT_PATH = 'edgeface_xxs.onnx'

    def export():
        # 1. Load Model
        # The 'get_model' function is defined in the repo's backbones/__init__.py
        # You might need to check the arguments required by the specific model version
        print(f"Loading {MODEL_NAME}...")
        model = get_model(MODEL_NAME)
        
        # Load weights
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        # Handle state_dict if wrapped in 'state_dict' key
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()

        # 2. Create Dummy Input
        # EdgeFace usually takes 112x112 input. Check repo for specific transform.
        # Shape: (Batch_Size, Channels, Height, Width) -> (1, 3, 112, 112)
        dummy_input = torch.randn(1, 3, 112, 112)

        # 3. Export
        print(f"Exporting to {OUTPUT_PATH}...")
        torch.onnx.export(
            model,
            dummy_input,
            OUTPUT_PATH,
            input_names=['input'],
            output_names=['embedding'],
            dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
            opset_version=12
        )
        print("Done!")

    if __name__ == '__main__':
        export()
    ```
4.  Run the script: `python export_onnx.py`.
5.  **Copy the generated `edgeface_xxs.onnx` file back to your Angular project's `src/assets/models/` folder.**

## 2. Get the Face Detector (SCRFD)

You also need a face detector to find faces before recognizing them. `EdgeFace` only recognizes *already cropped* faces.

1.  Download `scrfd_2.5g_kps.onnx` (found in many insightface repositories or model zoos).
2.  Place it in `src/assets/models/scrfd_2.5g_kps.onnx`.

**IF YOU CANNOT FIND SCRFD:**
This application is designed to fail gracefully or use a placeholder if the file is missing, but face detection will not work.
