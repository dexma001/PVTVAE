import os
import glob
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from models import PVTVAE
from dataset_pipeline import BONE_NAMES, BONE_MAP

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_extreme_demo():
    print("[발표용 데모 생성기] 극단적 클리핑 데이터 시뮬레이션을 시작합니다...")
    
    # 1. Trained Model
    model = PVTVAE(input_dim=147, latent_dim=64).to(DEVICE)
    ckpt_path = "checkpoints/pvtvae_epoch_100.pth"
    if not os.path.exists(ckpt_path):
        print(f"가중치 파일({ckpt_path})을 찾을 수 없습니다.")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # 2. Random Data
    pt_files = glob.glob("processed_motions/*.pt")
    target_file = np.random.choice(pt_files)
    print(target_file)
    original_motion = torch.load(target_file)[:30].clone() # 30프레임 추출

    # 3. 정교한 클리핑 버그 주입 (뼈 길이를 보존하는 강제 회전)
    demo_motion = original_motion.clone()
    
    # 10프레임 ~ 25프레임 구간에서 왼팔이 가슴을 깊숙이 파고들게 만듭니다.
    for f in range(10, 25):
        # 0도 -> 90도 -> 0도로 부드럽게 꺾이도록 사인(Sine) 곡선 적용
        ratio = np.sin((f - 10) / 15.0 * np.pi) 
        angle_deg = 15.0 * ratio # 15 degree

        pivot = demo_motion[f, BONE_MAP['LeftShoulder'], :3].numpy()
        chest = demo_motion[f, BONE_MAP['Chest'], :3].numpy()
        arm_vec = demo_motion[f, BONE_MAP['LeftLowerArm'], :3].numpy() - pivot
        target_vec = chest - pivot

        # 어깨를 기준으로 팔을 가슴 쪽으로 회전시키는 축 계산
        rot_axis = np.cross(arm_vec, target_vec)
        norm = np.linalg.norm(rot_axis)
        
        if norm > 1e-6:
            rot_axis = rot_axis / norm
            rot = R.from_rotvec(rot_axis * np.radians(angle_deg))

            # 어깨 하위 관절들(상박, 하박, 손)을 통째로 회전 
            for bone in ['LeftUpperArm', 'LeftLowerArm', 'LeftHand']:
                idx = BONE_MAP[bone]
                p = demo_motion[f, idx, :3].numpy()
                demo_motion[f, idx, :3] = torch.tensor(pivot + rot.apply(p - pivot))

                # 쿼터니언(회전 정보)도 함께 갱신하여 VAE가 헷갈리지 않게 함
                q = demo_motion[f, idx, 3:].numpy()
                demo_motion[f, idx, 3:] = torch.tensor((rot * R.from_quat(q)).as_quat())

    # 4. Inference Results
    input_tensor = demo_motion.unsqueeze(0).view(1, 30, -1).to(DEVICE)
    with torch.no_grad():
        recon_motion, _, _ = model(input_tensor)
    corrected_motion = recon_motion.view(30, 21, 7).cpu()

    # 5. 기존 시각화 툴이 읽을 수 있도록 동일한 이름으로 덮어쓰기
    os.makedirs("demo_results", exist_ok=True)
    torch.save(demo_motion, "demo_results/sample_original.pt")
    torch.save(corrected_motion, "demo_results/sample_corrected.pt")
    print("준비 완료! 이제 'python dataset_pipeline.py'를 실행하여 결과를 확인하세요.")

if __name__ == "__main__":
    create_extreme_demo()