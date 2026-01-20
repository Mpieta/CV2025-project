
import cv2
import torch
import numpy as np
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from Trainer import ExerciseDetector
from Dataset import RAW_LABELS, LABEL_MAP

CHECKPOINT_PATH = "lightning_logs/version_20/checkpoints/epoch=0-step=1932.ckpt"
LABEL_DIR = "dataset/labels"
INPUT_VIDEO = "dataset/dataset/anon/020_iwdfwze1_.mp4"
OUTPUT_VIDEO = "hires_output.mp4"
LIMIT_FRAMES = 5000
CLIP_LENGTH = 16
CAM_METHOD = "hirescam" # "gradcam"

ID_TO_NAME = {i: "No Exercise" if raw == -1 else f"Ex {raw}" for i, raw in enumerate(RAW_LABELS)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4321, 0.3946, 0.3764], std=[0.2280, 0.2214, 0.2169])
])

def apply_heatmap_to_frame(frame, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)

def get_cam_method(method_name, model, target_layers):
    method_name = method_name.lower()
    if method_name == "gradcam":
        return GradCAM(model=model, target_layers=target_layers)
    elif method_name == "hirescam":
        return HiResCAM(model=model, target_layers=target_layers)


def run_cam_visualization():
    print(f"Loading model from {CHECKPOINT_PATH}")
    model = ExerciseDetector.load_from_checkpoint(CHECKPOINT_PATH).to(device)
    model.eval()

    target_layers = [model.model.layer4[-1]]
    cam = get_cam_method(CAM_METHOD, model, target_layers)

    video_id = os.path.basename(INPUT_VIDEO).replace(".mp4", "")
    csv_path = os.path.join(LABEL_DIR, f"{video_id}.csv")
    df = pd.read_csv(csv_path, header=None)
    true_labels = df.set_index(0)[2].to_dict()

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frames_buffer = []
    raw_frames_buffer = []
    processed_count = 0
    current_heatmap = None
    current_pred_name = "None"
    current_confidence = 0.0

    print(f"Processing video: {INPUT_VIDEO}")
    print(f"Output will be saved to: {OUTPUT_VIDEO}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (LIMIT_FRAMES and processed_count >= LIMIT_FRAMES):
            break

        current_frame_num = processed_count + 1
        raw_true = true_labels.get(current_frame_num, -1)
        true_name = ID_TO_NAME.get(LABEL_MAP.get(raw_true, 0), "Unknown")

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(Image.fromarray(img_rgb))
        frames_buffer.append(img_tensor)
        raw_frames_buffer.append(frame.copy())

        if len(frames_buffer) == CLIP_LENGTH:
            input_tensor = torch.stack(frames_buffer, dim=1).unsqueeze(0).to(device)  # [B, C, T, H, W]

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                current_confidence, pred_id = torch.max(probs, dim=1)
                current_confidence = current_confidence.item() * 100
                pred_id = pred_id.item()
                current_pred_name = ID_TO_NAME[pred_id]

            # CAM computation
            targets = [ClassifierOutputTarget(pred_id)]
            heatmaps = cam(input_tensor=input_tensor, targets=targets)  # [B, H, W] or [B, T, H, W]
            if heatmaps.ndim == 4:  # 3D CAM output: [B, T, H, W]
                current_heatmap = heatmaps.mean(axis=1)[0]  # average over time
            else:
                current_heatmap = heatmaps[0]  # [H, W]

            frames_buffer.pop(0)
            raw_frames_buffer.pop(0)

        if current_heatmap is not None:
            frame = apply_heatmap_to_frame(frame, current_heatmap, alpha=0.4)

        cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)

        is_correct = (current_pred_name == true_name)
        match_color = (0, 255, 0) if is_correct else (0, 0, 255)
        if current_pred_name == "None":
            match_color = (255, 255, 255)

        cv2.putText(frame, f"TRUE: {true_name}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"PRED: {current_pred_name} ({current_confidence:.1f}%)", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, match_color, 2)

        out.write(frame)
        processed_count += 1

        if processed_count % 50 == 0:
            print(f"Processed {processed_count} frames...")

    cap.release()
    out.release()
    print(f"Done! Video saved as {OUTPUT_VIDEO}")


if __name__ == "__main__":
    run_cam_visualization()
