import os
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import Model as vlm_model

from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import json

class GroundedSam:
    def __init__(self, config_file, grounded_checkpoint, sam_version, sam_checkpoint, device):
        self.config_file = config_file
        self.grounded_checkpoint = grounded_checkpoint
        self.sam_version = sam_version
        self.sam_checkpoint = sam_checkpoint
        self.device = device

    def load_image(self, image_path):
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image

    def load_dino_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval().to(device)
        return model

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, device):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]
        filt_mask = logits.max(dim=1)[0] > box_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]
        tokenized = model.tokenizer(caption)
        pred_phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, model.tokenizer)
            for logit in logits_filt
        ]
        return boxes_filt, pred_phrases

    def run_grounding_dino(self, image_path, config_file, checkpoint_file, text_prompt, box_threshold, text_threshold, device):
        image_pil, image = load_image(image_path)
        model = load_dino_model(config_file, checkpoint_file, device)
        boxes, phrases = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device)
        del model
        torch.cuda.empty_cache()
        return image_pil, boxes, phrases

    def run_sam(self, image_pil, image_path, sam_version, sam_checkpoint, boxes, device):
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        boxes = boxes * torch.tensor([W, H, W, H]).to(device)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2]).to(device)
        masks, _, _ = predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=transformed_boxes.to(device), multimask_output=False
        )
        del predictor
        torch.cuda.empty_cache()
        return masks, image

    def show_mask(self, mask, ax, random_color=False):
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
        mask_image = mask.reshape(mask.shape[-2], mask.shape[-1], 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax, label):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        ax.text(x0, y0, label, color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))

    def save_mask_data(self, output_dir, mask_list, box_list, label_list):
        value = 0
        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0]] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
        
        json_data = [{'value': value, 'label': 'background'}]
        for label, box in zip(label_list, box_list):
            value += 1
            if '(' in label:
                name, logit = label.split('(')
                logit = logit[:-1]  # Remove the closing ')'
            else:
                name = label
                logit = "0.0"  # Default logit value if not present
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.cpu().numpy().tolist(),  # Ensure the box tensor is moved to the CPU
            })
        with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
            json.dump(json_data, f)

if __name__ == "__main__":
    config_file = "/home/sriram/vlm_ws/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "/home/sriram/vlm_ws/src/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    sam_version = "vit_b"
    sam_checkpoint = "/home/sriram/vlm_ws/src/segment-anything/models/vit_b.pth"
    image_path = "/home/sriram/vlm_ws/src/stretch_vlm_pkg/image_folder/dogs.jpg"
    text_prompt = "dogs ."
    output_dir = "/home/sriram/vlm_ws/src/stretch_vlm_pkg/image_folder"
    box_threshold = 0.35
    text_threshold = 0.25
    device = "cuda"

    os.makedirs(output_dir, exist_ok=True)

    image_pil, boxes, phrases = run_grounding_dino(image_path, config_file, grounded_checkpoint, text_prompt, box_threshold, text_threshold, device)
    masks, image = run_sam(image_pil, image_path, sam_version, sam_checkpoint, boxes, device)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes, phrases):
        show_box(box.cpu().numpy(), plt.gca(), label)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)

    save_mask_data(output_dir, masks, boxes, phrases)
