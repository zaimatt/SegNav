import os
import cv2
import re
import numpy as np

def create_inference_video(base_dir, output_video_path, fps=10):
    # 1. 获取并排序文件夹
    folders = [f for f in os.listdir(base_dir) if f.startswith("step_") and os.path.isdir(os.path.join(base_dir, f))]
    folders.sort(key=lambda x: int(x.split('_')[1]))
    
    if not folders:
        print("Error: No step folders found!")
        return

    video_writer = None
    print(f"Processing {len(folders)} frames...")

    for folder_name in folders:
        folder_path = os.path.join(base_dir, folder_name)
        img_path = os.path.join(folder_path, "image.jpg")
        npz_path = os.path.join(folder_path, "inference_result.npz")
        txt_path = os.path.join(folder_path, "decoder_inference_info.txt")
        
        if not all(os.path.exists(p) for p in [img_path, npz_path]):
            continue

        # --- 数据读取 ---
        # A. 读取图片
        img = cv2.imread(img_path)
        if img is None: continue

        # B. 从 NPZ 读取 qpos (Target Position)
        try:
            with np.load(npz_path) as data:
                qpos = data['qpos'] # 假设格式为 [x, y]
                target_str = f"[{qpos[0]:.2f}, {qpos[1]:.2f}]"
        except Exception as e:
            target_str = "[N/A]"
            print(f"Error loading npz in {folder_name}: {e}")

        # C. 从 TXT 读取未来 5 步速度并显示到左上角
        velocity_list = []
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                matches = re.findall(r"步骤 \d+: 线速度x=([-\d\.]+), 角速度z=([-\d\.]+)", content)
                for i in range(min(5, len(matches))):
                    v_x = float(matches[i][0])
                    ang_z = float(matches[i][1])
                    velocity_list.append((v_x, ang_z))
        except Exception as e:
            print(f"Error reading velocities: {e}")

        # --- 绘制与合成 ---
        if video_writer is None:
            h, w, _ = img.shape
            video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        font = cv2.FONT_HERSHEY_SIMPLEX
        line_height = 28
        target_text = f"Target : {target_str}"
        target_org = (20, 40)
        target_font_scale = 0.8
        target_thickness = 2
        (_, target_text_h), target_baseline = cv2.getTextSize(
            target_text, font, target_font_scale, target_thickness
        )
        # 将速度列表放在 Target 文本下方，避免重叠
        start_y = target_org[1] + target_baseline + target_text_h + 12
        # 绘制 Target Position (QPOS)
        cv2.putText(img, target_text, target_org, font, target_font_scale, (0, 0, 0), 4) # 黑色描边
        cv2.putText(img, target_text, target_org, font, target_font_scale, (255, 255, 255), 2) # 白色文字
        # 左上角显示未来 5 步速度 (英文, 两位小数)
        for idx, (v, w) in enumerate(velocity_list):
            y_pos = start_y + idx * line_height
            display_text = f"Step {idx}: v={v:.2f}, w={w:.2f}"
            cv2.putText(img, display_text, (15, y_pos), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, display_text, (15, y_pos), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        video_writer.write(img)

    if video_writer:
        video_writer.release()
        print(f"Success! Video saved to: {output_video_path}")

if __name__ == "__main__":
    BASE_DIR = "/Train/hjx/segnav/iros_video/dog_view/inf_img20250922_110546"
    OUT_FILE = "/Train/hjx/segnav/iros_video/dog_view/dog_inference_velocity_long_1.mp4"
    create_inference_video(BASE_DIR, OUT_FILE)