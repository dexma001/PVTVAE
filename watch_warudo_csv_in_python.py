import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

# 1. VMC 표준으로 추출된 CSV 데이터 로드
csv_file = 'Sample_data/Bandai_Dataset_csv_vmc/Bandai_Dataset_1_csv/dataset-1_bow_active_001.csv'  # 새로 뽑으신 파일명으로 변경하세요
df = pd.read_csv(csv_file)
frames = df['Frame'].unique()

# 2. 뼈대 계층 구조
parents = {
    'Hips': None,
    'Spine': 'Hips', 'Chest': 'Spine', 'UpperChest': 'Chest', 'Neck': 'UpperChest', 'Head': 'Neck',
    'LeftShoulder': 'UpperChest', 'LeftUpperArm': 'LeftShoulder', 'LeftLowerArm': 'LeftUpperArm', 'LeftHand': 'LeftLowerArm',
    'RightShoulder': 'UpperChest', 'RightUpperArm': 'RightShoulder', 'RightLowerArm': 'RightUpperArm', 'RightHand': 'RightLowerArm',
    'LeftUpperLeg': 'Hips', 'LeftLowerLeg': 'LeftUpperLeg', 'LeftFoot': 'LeftLowerLeg', 'LeftToes': 'LeftFoot',
    'RightUpperLeg': 'Hips', 'RightLowerLeg': 'RightUpperLeg', 'RightFoot': 'RightLowerLeg', 'RightToes': 'RightFoot'
}

# 🌟 3. 파이썬 속 가상 아바타의 뼈 길이 (Unity 미터 단위 기준)
# AI가 클리핑을 계산하려면 기준이 되는 '표준 체형'이 필요합니다.
base_offsets = {
    'Hips': [0, 0, 0], # Hips는 CSV 데이터의 위치를 따름
    'Spine': [0, 0.1, 0], 'Chest': [0, 0.1, 0], 'UpperChest': [0, 0.1, 0],
    'Neck': [0, 0.1, 0], 'Head': [0, 0.15, 0],
    'LeftShoulder': [-0.05, 0.1, 0], 'LeftUpperArm': [-0.15, 0, 0], 'LeftLowerArm': [-0.25, 0, 0], 'LeftHand': [-0.25, 0, 0],
    'RightShoulder': [0.05, 0.1, 0], 'RightUpperArm': [0.15, 0, 0], 'RightLowerArm': [0.25, 0, 0], 'RightHand': [0.25, 0, 0],
    'LeftUpperLeg': [-0.1, 0, 0], 'LeftLowerLeg': [0, -0.4, 0], 'LeftFoot': [0, -0.4, 0], 'LeftToes': [0, -0.1, 0.1],
    'RightUpperLeg': [0.1, 0, 0], 'RightLowerLeg': [0, -0.4, 0], 'RightFoot': [0, -0.4, 0], 'RightToes': [0, -0.1, 0.1]
}

valid_bones = df['BoneName'].unique()
bones_to_draw = [(parents[b], b) for b in valid_bones if parents.get(b) in valid_bones]

# 4. 프레임별 순방향 운동학 (Forward Kinematics)
all_positions = []
for f in frames:
    frame_data = df[df['Frame'] == f].set_index('BoneName')
    global_pos = {}
    global_rot = {}
    
    for bone in parents.keys():
        if bone not in frame_data.index: 
            continue
            
        row = frame_data.loc[bone]
        
        # 🌟 VMC 규칙: Hips만 CSV의 위치 사용, 나머지는 가상 아바타(base_offsets)의 뼈 길이를 사용!
        if bone == 'Hips':
            local_p = np.array([row['px'], row['pz'], row['py']]) # 좌표계 동기화
        else:
            # base_offsets(Unity 좌표계)를 Matplotlib(Python 좌표계)로 변환
            ux, uy, uz = base_offsets[bone]
            local_p = np.array([ux, uz, uy])
            
        # 회전값은 CSV의 순수 모션 데이터를 그대로 사용 (좌표계 역전 상쇄)
        lr = R.from_quat([-row['qx'], -row['qz'], -row['qy'], row['qw']])
        
        p_bone = parents[bone]
        if p_bone is None or p_bone not in global_pos:
            global_pos[bone] = local_p
            global_rot[bone] = lr
        else:
            # 부모 위치 + (부모 회전이 적용된 가상 뼈의 길이)
            global_pos[bone] = global_pos[p_bone] + global_rot[p_bone].apply(local_p)
            global_rot[bone] = global_rot[p_bone] * lr
            
    all_positions.append(global_pos)

# 5. 시각화 세팅 (Warudo 시뮬레이터 모드)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
lines = [ax.plot([], [], [], 'o-', lw=2.5, markersize=5, color='mediumseagreen')[0] for _ in bones_to_draw]

def update(frame_idx):
    pos = all_positions[frame_idx]
    for i, (p, c) in enumerate(bones_to_draw):
        if p in pos and c in pos:
            x = [pos[p][0], pos[c][0]]
            y = [pos[p][1], pos[c][1]]
            z = [pos[p][2], pos[c][2]]
            lines[i].set_data(x, y)
            lines[i].set_3d_properties(z)
            
    hip_pos = pos['Hips']
    cx, cy, cz = hip_pos[0], hip_pos[1], hip_pos[2]
    
    # 1미터 공간 고정 (트레드밀 효과)
    ax.set_xlim(cx - 1, cx + 1)
    ax.set_ylim(cy - 1, cy + 1)
    ax.set_zlim(cz - 1, cz + 1)
    ax.set_title(f'VMC Data Simulator - Frame: {frame_idx}')
    ax.view_init(elev=20, azim=45) 
    return lines

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=33, blit=False)
plt.show()