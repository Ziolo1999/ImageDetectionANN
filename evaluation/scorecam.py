import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image
from torchcam.methods import ScoreCAM
from torchvision import transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    """Overlay a colormapped mask on a background image

    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    >>> from torchcam.utils import overlay_mask
    >>> img = ...
    >>> cam = ...
    >>> overlay = overlay_mask(img, cam)

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay))[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img

def plot_scorecam(model, images_list, labels_list):
    fig, axes = plt.subplots(len(images_list), 4, figsize=(12, 16))

    for i in range(len(images_list)):
        img = images_list[i].permute(1,2,0)
        img = np.float32(img)
        axes[i,0].imshow(img, aspect='auto')
        axes[i,0].axis("off")
        axes[i,0].set_title(labels_list[i])
        for j, layer in enumerate([0,1,8]):
            img_tensor = images_list[i].unsqueeze(0).to(device)
            model = model.to(device)
            cam_extractor = ScoreCAM(model=model.model, target_layer='features.'+str(layer), batch_size=1)
            with torch.no_grad(): 
                out = model.model(img_tensor)

            activation_map = cam_extractor(out[0].argmax().item(), out)

            img_to_show = transforms.ToPILImage()(img_tensor.squeeze(0))
            cam_to_show = transforms.ToPILImage()(activation_map[0])

            overlay = overlay_mask(img_to_show, cam_to_show, alpha=0.5)
            axes[i,j+1].imshow(overlay, aspect='auto')
            axes[i,j+1].axis("off")
            axes[i,j+1].set_title("Layer: " + str(layer))
    model = model.cpu() 
    plt.subplots_adjust(wspace=0.1)