import cv2
import torch
import torch.nn.functional as F
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from Trainer import ExerciseDetector

CHECKPOINT_PATH = "lightning_logs/version_20/checkpoints/epoch=0-step=1932.ckpt"
LABEL_DIR = "dataset/labels"
INPUT_VIDEO = "dataset/dataset/anon/020_iwdfwze1_.mp4"
OUTPUT_VIDEO = "prototype_comparison.mp4"
LIMIT_FRAMES = 5000
CLIP_LENGTH = 16

# Mappings
RAW_LABELS = [-1, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
ID_TO_NAME = {i: "No Exercise" if raw == -1 else f"Ex {raw}" for i, raw in enumerate(RAW_LABELS)}
LABEL_MAP = {raw: i for i, raw in enumerate(RAW_LABELS)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4321, 0.3946, 0.3764], std=[0.2280, 0.2214, 0.2169])
])


@torch.no_grad()
def run_comparison_prototype():
    model = ExerciseDetector.load_from_checkpoint(CHECKPOINT_PATH).to(device).eval()

    video_id = os.path.basename(INPUT_VIDEO).replace(".mp4", "")
    csv_path = os.path.join(LABEL_DIR, f"{video_id}.csv")

    df = pd.read_csv(csv_path, header=None)
    true_labels = df.set_index(0)[2].to_dict()

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frames_buffer = []
    processed_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (LIMIT_FRAMES and processed_count >= LIMIT_FRAMES):
            break

        current_frame_num = processed_count + 1

        raw_true = true_labels.get(current_frame_num, -1) if true_labels else -1
        true_name = ID_TO_NAME.get(LABEL_MAP.get(raw_true, 0), "Unknown")

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(Image.fromarray(img_rgb))
        frames_buffer.append(img_tensor)

        pred_name = "None"
        confidence = 0.0

        if len(frames_buffer) == CLIP_LENGTH:
            input_tensor = torch.stack(frames_buffer, dim=1).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred_id = torch.max(probs, dim=1)

            pred_name = ID_TO_NAME[pred_id.item()]
            confidence = conf.item() * 100
            frames_buffer.pop(0)

        cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)

        is_correct = (pred_name == true_name)
        match_color = (0, 255, 0) if is_correct else (0, 0, 255)
        if pred_name == "None": match_color = (255, 255, 255)

        cv2.putText(frame, f"TRUE: {true_name}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, f"PRED: {pred_name} ({confidence:.1f}%)", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, match_color, 2)

        out.write(frame)
        processed_count += 1

    cap.release()
    out.release()
    print(f"Video saved as {OUTPUT_VIDEO}")


if __name__ == "__main__":
    run_comparison_prototype()