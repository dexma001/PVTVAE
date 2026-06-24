import os
import glob
import torch
import numpy as np
from models import PVTVAE
from dataset_pipeline import PARENTS, BONE_NAMES
from physics_module import DifferentiablePhysics

# 1. 환경 설정 및 디바이스 정의
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔥 추론 디바이스: {DEVICE}")

# 핵심 충돌 페어 정의 (검증용)
COLLIDING_PAIRS = [
    (('Hips', 'Chest'), ('LeftLowerArm', 'LeftHand')),   # 몸통 vs 왼팔
    (('Hips', 'Chest'), ('RightLowerArm', 'RightHand')), # 몸통 vs 오른팔
    (('LeftLowerArm', 'LeftHand'), ('RightLowerArm', 'RightHand')), # 왼팔 vs 오른팔
    (('LeftLowerLeg', 'LeftFoot'), ('RightLowerLeg', 'RightFoot'))  # 왼다리 vs 오른다리
]

def inference():
    # 2. 폴더 경로 탐색 (자동 감지)
    # 현재 실행 위치가 Transformer_Temp 내부인지, 아니면 Warudo_Send_Temp인지 파악합니다.
    checkpoint_dir = "../checkpoints" if os.path.exists("../checkpoints") else "checkpoints"
    motions_dir = "../processed_motions" if os.path.exists("../processed_motions") else "processed_motions"
    
    CHECKPOINT_PATH = os.path.join(checkpoint_dir, "pvtvae_epoch_100.pth")
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {CHECKPOINT_PATH}\n(폴더 위치를 확인해주세요!)")

    # 3. 학습한 Model
    model = PVTVAE(input_dim=147, latent_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval() # 평가(Inference) 모드로 전환
    print("성공: Trained model loaded.")

    physics_engine = DifferentiablePhysics(PARENTS, {b: 0.05 for b in BONE_NAMES}).to(DEVICE)

    # ==========================================
    # 4. 테스트 데이터 로드 및 전처리
    # ==========================================
    pt_files = glob.glob(os.path.join(motions_dir, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f" '{motions_dir}' 폴더에 .pt 파일이 없습니다.")
        
    # 랜덤으로 아무 파일이나 하나 골라서 테스트합니다 (원하시면 pt_files[0]으로 고정해도 됩니다)
    TEST_FILE_PATH = np.random.choice(pt_files) 
    print(f"🎬 테스트 대상 모션 파일: {os.path.basename(TEST_FILE_PATH)}")
    
    original_motion = torch.load(TEST_FILE_PATH)
    
    # 모델 입력 스펙([Batch=1, Seq=30, 147])에 맞추기 위해 첫 30프레임 추출 및 Flatten
    if original_motion.shape[0] < 30:
        raise ValueError(" 테스트 파일의 프레임 길이가 30보다 짧습니다.")
        
    input_sequence = original_motion[:30].unsqueeze(0) # [1, 30, 21, 7]
    input_flattened = input_sequence.view(1, 30, -1).to(DEVICE) # [1, 30, 147]

    # ==========================================
    # 5. AI 모델을 통한 모션 교정 (Inference)
    # ==========================================
    with torch.no_grad(): # 미분 계산을 꺼서 메모리를 절약하고 속도를 높입니다.
        recon_motion, _, _ = model(input_flattened)
        
    # 물리 연산 및 저장을 위해 다시 [1, 30, 21, 7] 형태로 복원
    recon_reshaped = recon_motion.view(1, 30, 21, 7)

    # ==========================================
    # 6. 물리 엔진 검증 (Before & After 충돌 오차 비교)
    # ==========================================
    input_sequence_cuda = input_sequence.to(DEVICE)
    
    loss_phys_before = physics_engine.get_collision_loss_from_tensor(input_sequence_cuda, COLLIDING_PAIRS)
    loss_phys_after = physics_engine.get_collision_loss_from_tensor(recon_reshaped, COLLIDING_PAIRS)

    print("\n==============================================")
    print("물리 기반 모션 교정 평가 (Physics Evaluation)")
    print("==============================================")
    print(f"교정 전 원본 충돌 수치 (Before): {loss_phys_before.item():.6f}")
    print(f"교정 후 AI 결과 충돌 수치 (After) : {loss_phys_after.item():.6f}")
    print("==============================================")

    # 7. 시각화 툴에서 로드할 수 있도록 결과 저장
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)
    
    save_path_orig = os.path.join(output_dir, "sample_original.pt")
    save_path_corr = os.path.join(output_dir, "sample_corrected.pt")
    
    # 배치를 떼어내고 원래 차원인 [30, 21, 7]로 CPU에 저장
    torch.save(input_sequence.squeeze(0).cpu(), save_path_orig)
    torch.save(recon_reshaped.squeeze(0).cpu(), save_path_corr)
    print(f" 결과 데이터가 '{output_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    inference()