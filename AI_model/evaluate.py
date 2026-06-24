import os
import torch
import numpy as np

# 기존 프로젝트 모듈 로드
from dataset_pipeline import PARENTS, BONE_NAMES, BONE_MAP
from physics_module import DifferentiablePhysics

# 평가에 사용할 충돌 페어
COLLIDING_PAIRS = [
    (('Hips', 'Chest'), ('LeftLowerArm', 'LeftHand')),   
    (('Hips', 'Chest'), ('RightLowerArm', 'RightHand')), 
    (('LeftLowerArm', 'LeftHand'), ('RightLowerArm', 'RightHand')), 
    (('LeftLowerLeg', 'LeftFoot'), ('RightLowerLeg', 'RightFoot'))  
]

def calculate_bone_length_error(motion):
    """지표 1: 뼈 길이 유지 오차 (고무줄 팔 검증)"""
    bone_lengths = []
    for child, parent in PARENTS.items():
        if parent is not None:
            c_idx, p_idx = BONE_MAP[child], BONE_MAP[parent]
            c_pos, p_pos = motion[:, c_idx, :3], motion[:, p_idx, :3]
            lengths = torch.norm(c_pos - p_pos, dim=-1)
            bone_lengths.append(torch.std(lengths).item())
    return np.mean(bone_lengths) * 100.0

def calculate_detailed_metrics(orig_motion, corr_motion):
    """지표 2: MPJPE 및 정밀 분석"""
    pos_orig = orig_motion[:, :, :3]
    pos_corr = corr_motion[:, :, :3]
    
    # 각 프레임, 각 관절별 유클리드 거리 (cm 변환)
    errors = torch.norm(pos_orig - pos_corr, dim=-1) * 100.0  # [Frames, Joints]
    
    total_mpjpe = errors.mean().item()
    joint_errors = errors.mean(dim=0)
    sorted_joints_idx = torch.argsort(joint_errors, descending=True)
    
    frame_errors = errors.mean(dim=1)
    peak_frame = torch.argmax(frame_errors).item()
    peak_error = frame_errors[peak_frame].item()
    
    max_joint_idx_at_peak = torch.argmax(errors[peak_frame]).item()
    max_movement = errors[peak_frame, max_joint_idx_at_peak].item()
    
    return total_mpjpe, joint_errors, sorted_joints_idx, peak_frame, peak_error, max_joint_idx_at_peak, max_movement

def evaluate():
    print("⏳ AI 모션 교정 통합 성능 평가를 시작합니다...")
    
    orig_path = "demo_results/sample_original.pt"
    corr_path = "demo_results/sample_corrected.pt"
    
    if not os.path.exists(orig_path) or not os.path.exists(corr_path):
        print("❌ 평가할 데이터를 찾을 수 없습니다.")
        return
        
    motion_orig = torch.load(orig_path)
    motion_corr = torch.load(corr_path)
    orig_batch, corr_batch = motion_orig.unsqueeze(0), motion_corr.unsqueeze(0)
    
    # 1. 물리 엔진 평가
    physics_engine = DifferentiablePhysics(PARENTS, {b: 0.05 for b in BONE_NAMES})
    col_loss_orig = physics_engine.get_collision_loss_from_tensor(orig_batch, COLLIDING_PAIRS).item()
    col_loss_corr = physics_engine.get_collision_loss_from_tensor(corr_batch, COLLIDING_PAIRS).item()
    
    bone_err_orig = calculate_bone_length_error(motion_orig)
    bone_err_corr = calculate_bone_length_error(motion_corr)
    
    # 2. 정밀 오차 평가
    total_mpjpe, joint_errs, sorted_j_idx, p_frame, p_error, max_j_idx, max_move = calculate_detailed_metrics(motion_orig, motion_corr)
    
    # --- 결과 출력 ---
    print("\n" + "="*60)
    print("🏆 물리 기반 모션 교정(PVTVAE) 통합 성적표")
    print("="*60)
    
    print("\n[1] 물리적 무결성 (Physical Plausibility)")
    print(f"  ▶ 충돌 위험도 (Collision Depth)")
    print(f"     - Before : {col_loss_orig:.6f}")
    print(f"     - After  : {col_loss_corr:.6f}")
    if col_loss_orig > 0:
        print(f"     ✅ 충돌이 {((col_loss_orig - col_loss_corr) / col_loss_orig * 100):.1f}% 감소했습니다!")
        
    print(f"\n  ▶ 뼈 길이 변동성 (Bone Length Jitter, 낮을수록 좋음)")
    print(f"     - Before : {bone_err_orig:.4f} cm")
    print(f"     - After  : {bone_err_corr:.4f} cm")
    
    print("\n" + "-"*60)
    print("[2] 모션 보존 및 정밀 분석 (Kinematic Accuracy)")
    print(f"  ▶ 전체 평균 오차 (MPJPE): {total_mpjpe:.2f} cm")
    
    print("\n  ▶ 관절별 이동 거리 Top 5 (AI가 집중적으로 고친 곳)")
    for i in range(5):
        j_idx = sorted_j_idx[i].item()
        print(f"     {i+1}위: {BONE_NAMES[j_idx]:<15} -> 평균 {joint_errs[j_idx].item():.2f} cm 이동")
        
    print("\n  ▶ 결정적 교정 순간 (Peak Frame Analysis)")
    print(f"     - 개입 최대 프레임 : {p_frame}번 프레임")
    print(f"     🚨 [핵심 팩트] 해당 프레임에서 '{BONE_NAMES[max_j_idx]}' 관절이")
    print(f"        물리 법칙을 지키기 위해 무려 **{max_move:.2f} cm** 이동했습니다!")
    print("="*60)

if __name__ == "__main__":
    evaluate()