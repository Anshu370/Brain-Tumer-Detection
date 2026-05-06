import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # ✅ Forward hook (activations)
        self.target_layer.register_forward_hook(self._forward_hook)

        # ✅ Backward hook (SAFE version)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        """
        input_tensor: shape (1, C, H, W)
        class_idx: int
        """

        # Ensure gradients are enabled
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        output = self.model(input_tensor)

        # Zero grads
        self.model.zero_grad()

        # Backward (only target class)
        loss = output[:, class_idx]
        loss.backward()

        # Get stored activations & gradients
        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0] # (C, H, W)

        # ✅ Global Average Pooling (weights)
        weights = gradients.mean(dim=(1, 2))  # (C)

        # ✅ Weighted sum
        cam = torch.zeros(activations.shape[1:], device=input_tensor.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ✅ ReLU
        cam = F.relu(cam)

        # ✅ Normalize safely
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.cpu().numpy()