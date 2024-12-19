import torch
from core import model as md
import coremltools as ct

# Path to your checkpoint file
ckpt_path = "/Users/ricoqi/Desktop/All/Dabble/MobileFaceNet_Pytorch/model/best/068.ckpt"

# 1. Load the PyTorch model
model = md.MobileFacenet()
model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['net_state_dict'])
model.eval()

# 2. Provide a dummy input to trace the model
# Adjust this shape according to your model's expected input dimensions.
# Common inputs for face-related models might be (1, 3, 112, 112)
dummy_input = torch.randn(1, 3, 112, 112)

# 3. Trace or script the model for Core ML conversion
traced_model = torch.jit.trace(model, dummy_input)

# 4. Convert the PyTorch model to Core ML (.mlmodel)
# Note: If your model expects images, you can specify ImageType for better integration.
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=dummy_input.shape)]
)

# 5. Save the resulting .mlmodel file
mlmodel.save("MobileFacenet.mlpackage")

print("Conversion successful! The model has been saved as 'MobileFacenet.mlmodel'.")
