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
        txt_path = os.path.join(folder_path, "decoder_inference_info.txt") # 仅用于获取 action
        
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

        # C. 从 TXT 读取角速度判断 Action
        action_str = "forward"
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                step0_match = re.search(r"步骤 0:.*?角速度z=([-\d\.]+)", content)
                if step0_match:
                    ang_z = float(step0_match.group(1))
                    if ang_z > 0.2: action_str = "turn_left"
                    elif ang_z < -0.2: action_str = "turn_right"
        except:
            pass

        # --- 绘制与合成 ---
        if video_writer is None:
            h, w, _ = img.shape
            video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # 样式定义
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 绘制 Target Position (QPOS)
        cv2.putText(img, f"Target : {target_str}", (20, 40), font, 0.8, (0, 0, 0), 4) # 黑色描边
        cv2.putText(img, f"Target : {target_str}", (20, 40), font, 0.8, (255, 255, 255), 2) # 白色文字
        
        # 绘制 Action
        color = (0, 255, 0) # Green
        if "left" in action_str: color = (0, 255, 255) # Yellow
        elif "right" in action_str: color = (0, 0, 255) # Red
            
        cv2.putText(img, f"Action: {action_str}", (20, 80), font, 0.8, (0, 0, 0), 4)
        cv2.putText(img, f"Action: {action_str}", (20, 80), font, 0.8, color, 2)

        video_writer.write(img)

    if video_writer:
        video_writer.release()
        print(f"Success! Video saved to: {output_video_path}")

if __name__ == "__main__":
    BASE_DIR = "/Train/hjx/segnav/iros_video/dog_view/inf_img20250922_110546"
    OUT_FILE = "/Train/hjx/segnav/iros_video/dog_view/dog_inference_qpos_long.mp4"
    create_inference_video(BASE_DIR, OUT_FILE)