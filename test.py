import cv2
import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from Trainer import ExerciseDetector

# --- CONFIGURATION ---
CHECKPOINT_PATH = "lightning_logs/version_34/checkpoints/epoch=1-step=5664.ckpt"
LABEL_DIR = "dataset/labels"
INPUT_VIDEO = "dataset/dataset/anon/016_fd0hyvd1.mp4"
OUTPUT_VIDEO = "analysis_comparison_final.mp4"
OUTPUT_CSV = "prediction_metrics.csv"
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
def run_comparison_analysis():
    # Load model from checkpoint
    model = ExerciseDetector.load_from_checkpoint(CHECKPOINT_PATH, model_name="mobilenet_v2_3d").to(device).eval()

    video_id = os.path.basename(INPUT_VIDEO).replace(".mp4", "")
    csv_path = os.path.join(LABEL_DIR, f"{video_id}.csv")

    # Load Ground Truth
    df = pd.read_csv(csv_path, header=None)
    true_labels = df.set_index(0)[2].to_dict()

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize Video Writer
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frames_buffer = []
    processed_count = 0
    results_data = []
    prev_probs = None  # To track frame-to-frame stability

    print(f"Processing video: {INPUT_VIDEO}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (LIMIT_FRAMES and processed_count >= LIMIT_FRAMES):
            break

        current_frame_num = processed_count + 1
        raw_true = true_labels.get(current_frame_num, -1)
        true_idx = LABEL_MAP.get(raw_true, -1)
        true_name = ID_TO_NAME.get(true_idx, "Unknown")

        # Convert and transform for model
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(Image.fromarray(img_rgb))
        frames_buffer.append(img_tensor)

        pred_name = "Buffering..."
        confidence = 0.0
        kl_div_error = np.nan
        stability_kl = np.nan
        is_correct = False

        if len(frames_buffer) == CLIP_LENGTH:
            input_tensor = torch.stack(frames_buffer, dim=1).unsqueeze(0).to(device)
            logits = model(input_tensor)

            probs = F.softmax(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=1)

            conf, pred_id = torch.max(probs, dim=1)
            pred_name = ID_TO_NAME[pred_id.item()]
            confidence = conf.item() * 100
            is_correct = (pred_name == true_name)

            # --- KL Divergence Analysis ---
            # 1. Error (Target vs Pred)
            if true_idx != -1:
                target_dist = torch.zeros_like(probs)
                target_dist[0, true_idx] = 1.0
                kl_div_error = F.kl_div(log_probs, target_dist, reduction='batchmean').item()

            # 2. Jitter (Previous Pred vs Current Pred)
            if prev_probs is not None:
                stability_kl = F.kl_div(log_probs, prev_probs, reduction='batchmean').item()

            prev_probs = probs.detach()
            frames_buffer.pop(0)

        # Drawing UI
        cv2.rectangle(frame, (0, 0), (width, 150), (0, 0, 0), -1)
        match_color = (0, 255, 0) if is_correct else (0, 0, 255)
        if pred_name == "Buffering...": match_color = (255, 255, 255)

        cv2.putText(frame, f"TRUE: {true_name}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"PRED: {pred_name} ({confidence:.1f}%)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    match_color, 2)

        kl_text = f"KL Err: {kl_div_error:.4f} | Jitter: {stability_kl:.4f}"
        cv2.putText(frame, kl_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        out.write(frame)

        # Log metrics for CSV
        if pred_name != "Buffering..." and true_idx != -1:
            results_data.append({
                "frame": current_frame_num,
                "kl_error": kl_div_error,
                "jitter": stability_kl,
                "confidence": confidence,
                "correct": 1 if is_correct else 0
            })

        processed_count += 1

    cap.release()
    out.release()
    print(f"Video output saved to: {OUTPUT_VIDEO}")

    # Summary Report
    if results_data:
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(OUTPUT_CSV, index=False)
        print("\n" + "=" * 30)
        print("SUMMARY STATISTICS")
        print("=" * 30)
        print(f"Accuracy:      {df_results['correct'].mean() * 100:.2f}%")
        print(f"Avg KL Error:  {df_results['kl_error'].mean():.4f}")
        print(f"Avg Jitter:    {df_results['jitter'].mean():.4f}")
        print(f"Data saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_comparison_analysis()