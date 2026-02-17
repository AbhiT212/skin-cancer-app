import torch
import numpy as np
from PIL import Image
import io
import cv2
import base64
from config import CONFIG, IDX_TO_CLASS
from model import SkinCancerHybrid_Pro
from utils import get_inference_transforms, process_metadata

print(f"Loading model on {CONFIG['device']}...")
model = SkinCancerHybrid_Pro(
    num_classes=CONFIG['num_classes'],
    num_meta_features=CONFIG['num_meta_features'],
)
try:
    model.load_state_dict(
        torch.load("weights/best_model.pth", map_location=CONFIG['device'])
    )
    model.to(CONFIG['device'])
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("WARNING: 'weights/best_model.pth' not found. Running without weights.")


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
    img_orig  = image_tensor.unsqueeze(0).to(CONFIG['device'])
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

    Parameters
    ----------
    image_bytes   : BytesIO — raw image bytes
    age           : int     — patient age
    sex           : str     — 'male' | 'female'
    localization  : str     — body location string

    Returns
    -------
    dict with keys: top_prediction, top_confidence, margin, is_uncertain,
                    classes, probabilities, gradcam_base64
    """
    image        = Image.open(image_bytes).convert("RGB")
    image_tensor = get_inference_transforms()(image)
    meta_tensor  = process_metadata(age, sex, localization)

    # 1. TTA prediction (no gradients needed here)
    probs          = predict_tta(model, image_tensor, meta_tensor)
    probs          = probs.squeeze().cpu().numpy()
    sorted_indices = np.argsort(probs)[::-1]
    top_1_idx      = sorted_indices[0]

    # 2. Grad-CAM on the last conv block of the EfficientNet stream
    target_layer = model.stream_b.backbone[-1]
    grad_cam     = CustomGradCAM(model, target_layer)

    img_batch  = image_tensor.unsqueeze(0).to(CONFIG['device'])
    img_batch.requires_grad_(True)
    meta_batch = meta_tensor.unsqueeze(0).to(CONFIG['device'])

    cam_heatmap = grad_cam.generate(img_batch, meta_batch, top_1_idx)

    # 3. Overlay heatmap on original image
    img_arr     = np.array(image.resize((CONFIG['img_size'], CONFIG['img_size'])))
    cam_resized = cv2.resize(cam_heatmap, (img_arr.shape[1], img_arr.shape[0]))

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = (heatmap_colored * 0.4 + img_arr * 0.6).astype(np.uint8)

    # 4. Encode overlay to base64 for the UI
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