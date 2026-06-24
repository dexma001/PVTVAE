import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Modules
from dataset_pipeline import BandaiMotionDataset, PARENTS, BONE_RADII
from models import PVTVAE
from physics_module import DifferentiablePhysics

# 1. 하이퍼파라미터 및 환경 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔥 학습 디바이스: {DEVICE}")

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# 전처리된 데이터 경로
PT_DIR = "processed_motions"

# 2. 핵심 충돌 페어 정의 (Curriculum Learning)
# 전신을 모두 검사하면 느려지므로, 가장 잘 충돌하는 핵심 그룹만 묶어줍니다.
COLLIDING_PAIRS = [
    (('Hips', 'Chest'), ('LeftLowerArm', 'LeftHand')),   # 몸통 vs 왼팔
    (('Hips', 'Chest'), ('RightLowerArm', 'RightHand')), # 몸통 vs 오른팔
    (('LeftLowerArm', 'LeftHand'), ('RightLowerArm', 'RightHand')), # 왼팔 vs 오른팔
    (('LeftLowerLeg', 'LeftFoot'), ('RightLowerLeg', 'RightFoot'))  # 왼다리 vs 오른다리
]

# 3. 모델, 데이터, 물리 엔진 초기화
dataset = BandaiMotionDataset(PT_DIR, seq_len=30)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # 윈도우 에러 방지용 0

model = PVTVAE(input_dim=147, latent_dim=64).to(DEVICE)
physics_engine = DifferentiablePhysics(PARENTS, BONE_RADII).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. 메인 학습 루프
def train():
    os.makedirs("checkpoints", exist_ok=True)
    
    log_file = "log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("🚀 Training Started\n")
        f.write("="*50 + "\n")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_recon_loss = 0
        total_kl_loss = 0
        total_phys_loss = 0
        
        # 단계적 물리 제약 (Curriculum Learning)
        # 처음 20 에폭은 동작만 배우고, 그 이후부터 물리 법칙(충돌 회피)을 주입
        lambda_phys = 0.0 if epoch <= 20 else min(0.1, (epoch - 20) * 0.005)
        beta_kl = 0.01 # VAE 정규화 가중치
        
        for batch_data in dataloader:
            # batch_data: [Batch, 30, 21, 7] -> [Batch, 30, 147] 로 변환
            batch_data = batch_data.view(batch_data.size(0), batch_data.size(1), -1).to(DEVICE)
            
            optimizer.zero_grad()
            
            # 1. Forward
            recon_motion, mu, logvar = model(batch_data)
            
            # 2. Loss 계산
            loss_recon = nn.MSELoss()(recon_motion, batch_data)
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # 물리 엔진을 위해 다시 [Batch, 30, 21, 7] 형태로 해석 (위치값 추출)
            # 여기서는 편의상 모델 출력을 딕셔너리로 변환하여 물리 엔진에 전달
            # (실제 구조에 맞춰 뼈대 인덱싱 필요)
            recon_reshaped = recon_motion.view(-1, 30, 21, 7)
            
            # ---- [주의] 물리 Loss 계산 로직 연동 ----
            # 모델 출력(recon_reshaped)을 DifferentiablePhysics가 읽을 수 있도록
            # global_pos 딕셔너리 형태로 래핑하는 과정이 필요합니다.
            loss_phys = torch.tensor(0.0, device=DEVICE)
            if lambda_phys > 0:
                # 이 부분은 데이터 구조에 맞게 BONE_MAP을 통한 인덱스 추출로 교체
                loss_phys = physics_engine.get_collision_loss_from_tensor(recon_reshaped, COLLIDING_PAIRS)
            # ----------------------------------------
            
            # 3. 역전파 및 최적화
            loss = loss_recon + (beta_kl * loss_kl) + (lambda_phys * loss_phys)
            loss.backward()
            
            # 💥 안전벨트 추가: 기울기(Gradient)가 폭발하지 않도록 최대치 제한
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_recon_loss += loss_recon.item()
            total_kl_loss += loss_kl.item()
            total_phys_loss += loss_phys.item()
            
        # 에폭 결과 출력
        num_batches = len(dataloader)
        log_msg = (f"Epoch [{epoch}/{EPOCHS}] "
                   f"Recon: {total_recon_loss/num_batches:.4f} | "
                   f"KL: {total_kl_loss/num_batches:.4f} | "
                   f"Phys: {total_phys_loss/num_batches:.4f} (λ={lambda_phys:.3f})")
        
        # 터미널 화면에 출력
        print(log_msg)
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
        
        # 모델 체크포인트 저장 (10 에폭마다)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/pvtvae_epoch_{epoch}.pth")
            ckpt_msg = f"💾 Checkpoint saved: epoch {epoch}"
            print(ckpt_msg)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(ckpt_msg + "\n")

if __name__ == "__main__":
    train()