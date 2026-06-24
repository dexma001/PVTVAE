import torch
from physics_module import DifferentiablePhysics

# 1. 환경 설정 (캡슐 생성을 위한 최소한의 부모-자식 관계)
PARENTS = {
    'Hips': None,
    'Chest': 'Hips',
    'LeftShoulder': 'Chest',
    'LeftHand': 'LeftShoulder'
}
# 뼈대의 두께(반경) 설정
BONE_RADII = {'Chest': 0.1, 'LeftHand': 0.05} 

physics_engine = DifferentiablePhysics(PARENTS, BONE_RADII)

# 2. 충돌 검사 쌍 (수정된 부분!)
# 선분 1: Hips ~ Chest (몸통 캡슐)
# 선분 2: LeftShoulder ~ LeftHand (왼팔 캡슐)
colliding_pairs = [ (('Hips', 'Chest'), ('LeftShoulder', 'LeftHand')) ]

# 3. 테스트용 데이터 생성
# A. Safe (안전): 팔을 바깥으로 뻗은 상태
pos_safe = {
    'Hips': torch.tensor([0.0, 0.0, 0.0]),
    'Chest': torch.tensor([0.0, 0.0, 0.5]),
    'LeftShoulder': torch.tensor([0.5, 0.0, 0.5]),
    'LeftHand': torch.tensor([0.5, 0.0, 0.0]) # 몸통(0.0)에서 X축으로 0.5만큼 떨어짐
}

# B. Collision (충돌): 손이 가슴 영역을 파고든 상태
pos_col = {
    'Hips': torch.tensor([0.0, 0.0, 0.0]),
    'Chest': torch.tensor([0.0, 0.0, 0.5]),
    'LeftShoulder': torch.tensor([0.5, 0.0, 0.5]),
    'LeftHand': torch.tensor([0.05, 0.0, 0.25]) # X축 0.05 위치로 몸통을 뚫고 들어옴!
}

# 4. 테스트 실행
# (주의: 만약 physics_module 내부에서 디바이스 오류가 난다면,
# DifferentiablePhysics 내부의 torch.zeros 등 텐서 생성 코드에 device='cpu'를 맞춰주어야 할 수 있습니다.)
loss_safe = physics_engine.get_collision_loss(pos_safe, colliding_pairs)
loss_col = physics_engine.get_collision_loss(pos_col, colliding_pairs)

print(f"Safe Loss: {loss_safe.item():.6f}")
print(f"Collision Loss: {loss_col.item():.6f}")

if loss_col > loss_safe:
    print("✅ 성공: 물리 엔진이 캡슐 충돌을 정확히 감지했습니다!")
else:
    print("❌ 실패: 물리 페널티가 제대로 계산되지 않았습니다.")