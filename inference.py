import torch
import numpy as np
from PIL import Image
import io
import cv2
import base64
from config import CONFIG, IDX_TO_CLASS
from model import SkinCancerHybrid_Pro
from utils import get_inference_transforms, process_metadata

# Silence the PyTorch NNPACK warning for Cloud Run virtual CPUs
torch.backends.nnpack.enabled = False

print(f"Loading model on {CONFIG['device']}...")
model = SkinCancerHybrid_Pro(
    num_classes=CONFIG['num_classes'],
    num_meta_features=CONFIG['num_meta_features'],
)

# 1. Load weights from the live Cloud Storage bucket volume mount
try:
    model.load_state_dict(
        torch.load("weights/skin-cancer-app/best_model.pth", map_location=CONFIG['device'])
    )
    print("Model loaded successfully from the live bucket volume.")
except Exception as e:
    print(f"CRITICAL WARNING: 'weights/skin-cancer-app/best_model.pth' not found. Error: {e}")

# 2. THE FIX: These must be OUTSIDE the try block so they are guaranteed to run!
model.to(CONFIG['device'])
model.eval()


class CustomGradCAM:
    """Lightweight Grad-CAM tailored for the hybrid image + metadata model."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, meta_tensor, target_class):
        self.model.zero_grad()

        logits = self.model(image_tensor, meta_tensor)
        score  = logits[0, target_class]
        score.backward()

        activations = self.activations.detach().cpu().numpy()[0]
        gradients   = self.gradients.detach().cpu().numpy()[0]
        weights     = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  # ReLU

        if np.max(cam) != 0:
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)

        return cam


def predict_tta(model, image_tensor, meta_tensor):
    """4-view test-time augmentation: original, h-flip, v-flip, 90° rotation."""
    
    # 3. Double-enforce evaluation mode just to be safe
    model.eval()
    
    img_orig   = image_tensor.unsqueeze(0).to(CONFIG['device'])
    meta_batch = meta_tensor.unsqueeze(0).to(CONFIG['device'])

    views = [
        img_orig,
        torch.flip(img_orig, [3]),
        torch.flip(img_orig, [2]),
        torch.rot90(img_orig, 1, [2, 3]),
    ]

    probs_sum = torch.zeros(1, CONFIG['num_classes']).to(CONFIG['device'])
    with torch.no_grad():
        for view in views:
            probs_sum += torch.nn.functional.softmax(
                model(view, meta_batch), dim=1
            )

    return probs_sum / len(views)


def run_inference(image_bytes, age, sex, localization):
    """
    Run full inference pipeline.
    """
    # 1. Load Original RGB Image (We keep this for the UI Visualization)
    image_rgb = Image.open(image_bytes).convert("RGB")
    
    # 2. THE COLOR FIX: Convert a copy to BGR for the model (Matches Kaggle cv2 training)
    image_bgr_arr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
    image_bgr = Image.fromarray(image_bgr_arr)
    
    # 3. Pass the BGR image to your transforms
    image_tensor = get_inference_transforms()(image_bgr)
    meta_tensor  = process_metadata(age, sex, localization)

    # 4. TTA prediction
    probs          = predict_tta(model, image_tensor, meta_tensor)
    probs          = probs.squeeze().cpu().numpy()
    sorted_indices = np.argsort(probs)[::-1]
    top_1_idx      = sorted_indices[0]

    # 5. Grad-CAM on the last conv block of the EfficientNet stream
    target_layer = model.stream_b.backbone[-1]
    grad_cam     = CustomGradCAM(model, target_layer)

    img_batch  = image_tensor.unsqueeze(0).to(CONFIG['device'])
    img_batch.requires_grad_(True)
    meta_batch = meta_tensor.unsqueeze(0).to(CONFIG['device'])

    cam_heatmap = grad_cam.generate(img_batch, meta_batch, top_1_idx)

    # 6. Overlay heatmap on original RGB image (So the UI doesn't look blue!)
    img_arr     = np.array(image_rgb.resize((CONFIG['img_size'], CONFIG['img_size'])))
    cam_resized = cv2.resize(cam_heatmap, (img_arr.shape[1], img_arr.shape[0]))

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = (heatmap_colored * 0.4 + img_arr * 0.6).astype(np.uint8)

    # 7. Encode overlay to base64 for the UI
    pil_overlay = Image.fromarray(superimposed)
    buf         = io.BytesIO()
    pil_overlay.save(buf, format="JPEG")
    cam_base64  = (
        "data:image/jpeg;base64,"
        + base64.b64encode(buf.getvalue()).decode("utf-8")
    )

    margin = float(probs[top_1_idx] - probs[sorted_indices[1]])

    return {
        "top_prediction": IDX_TO_CLASS[top_1_idx],
        "top_confidence": float(probs[top_1_idx]),
        "margin":         margin,
        "is_uncertain":   bool(margin < 0.15),
        "classes":        list(IDX_TO_CLASS.values()),
        "probabilities":  [float(probs[i]) for i in range(len(IDX_TO_CLASS))],
        "gradcam_base64": cam_base64,
    }