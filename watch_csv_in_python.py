import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

# 1. 데이터 로드
csv_file = r"Sample_Data/Bandai_Dataset_csv_modi_tot/Bandai_Dataset_2_csv/dataset-2_run_feminine_029.csv"

df = pd.read_csv(csv_file)
frames = df['Frame'].unique()

# 2. 뼈대 계층 구조 (추출된 21개의 뼈에 맞춘 최종 버전)
parents = {
    'Hips': None,
    'Spine': 'Hips', 'Chest': 'Spine', 
    'Neck': 'Chest', 'Head': 'Neck',
    # 팔 (Shoulder 포함)
    'LeftShoulder': 'Chest', 'LeftUpperArm': 'LeftShoulder', 'LeftLowerArm': 'LeftUpperArm', 'LeftHand': 'LeftLowerArm',
    'RightShoulder': 'Chest', 'RightUpperArm': 'RightShoulder', 'RightLowerArm': 'RightUpperArm', 'RightHand': 'RightLowerArm',
    # 다리 (Toes 포함)
    'LeftUpperLeg': 'Hips', 'LeftLowerLeg': 'LeftUpperLeg', 'LeftFoot': 'LeftLowerLeg', 'LeftToes': 'LeftFoot',
    'RightUpperLeg': 'Hips', 'RightLowerLeg': 'RightUpperLeg', 'RightFoot': 'RightLowerLeg', 'RightToes': 'RightFoot'
}

# 유효한 뼈대만 필터링하여 선으로 연결할 쌍 생성
valid_bones = df['BoneName'].unique()
bones_to_draw = [(parents[b], b) for b in valid_bones if parents.get(b) in valid_bones]

# 3. 프레임별 순방향 운동학 (FK) 계산
all_positions = []
for f in frames:
    frame_data = df[df['Frame'] == f].set_index('BoneName')
    global_pos = {}
    global_rot = {}
    
    for bone in parents.keys():
        if bone not in frame_data.index: 
            continue
        
        row = frame_data.loc[bone]
        
        # 🌟 핵심 1: Unity(왼손/Y-up) -> Python(오른손/Z-up) 축 변환
        local_p = np.array([row['px'], -row['pz'], -row['py']])
        # 🌟 핵심 2: 회전 방향 역전 상쇄
        lr = R.from_quat([-row['qx'], -row['qz'], -row['qy'], row['qw']])
        
        # 🌟 핵심 3: 스케일 원상복구 (Hips 제외)
        if bone != 'Hips':
            local_p *= 100.0  
            
        p_bone = parents[bone]
        if p_bone is None or p_bone not in global_pos:
            global_pos[bone] = local_p
            global_rot[bone] = lr
        else:
            # 부모의 회전값을 반영하여 자식의 현재 위치를 계산
            global_pos[bone] = global_pos[p_bone] + global_rot[p_bone].apply(local_p)
            global_rot[bone] = global_rot[p_bone] * lr
            
    all_positions.append(global_pos)

# 4. Matplotlib 3D 뷰어 세팅
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
# 선 굵기와 관절 포인트(마커) 디자인 설정
lines = [ax.plot([], [], [], 'o-', lw=2.5, markersize=5, color='royalblue')[0] for _ in bones_to_draw]

def update(frame_idx):
    pos = all_positions[frame_idx]
    for i, (p, c) in enumerate(bones_to_draw):
        if p in pos and c in pos:
            x = [pos[p][0], pos[c][0]]
            y = [pos[p][1], pos[c][1]]
            z = [pos[p][2], pos[c][2]]
            lines[i].set_data(x, y)
            lines[i].set_3d_properties(z)
            
    # 사람 키(약 2m)에 맞춘 카메라 공간 고정
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.set_title(f'3D Motion Playback - Frame: {frame_idx}')
    
    # 45도 측면에서 바라보는 기본 앵글
    ax.view_init(elev=20, azim=45) 
    return lines

# 애니메이션 재생 (약 30 FPS 속도)
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=33, blit=False)
plt.show()