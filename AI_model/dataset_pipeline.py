import os
import glob
import torch
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import Dataset, DataLoader
import itertools

# 0. 설정 및 상수
MODE = "VISUALIZE"  # "PREPROCESS" 또는 "VISUALIZE"
SEQ_LEN = 30
CSV_DIR = r"Sample_Data/Bandai_Dataset_csv_modi_tot"
PT_DIR = "processed_motions"

PARENTS = {
    'Hips': None, 'Spine': 'Hips', 'Chest': 'Spine', 'Neck': 'Chest', 'Head': 'Neck',
    'LeftShoulder': 'Chest', 'LeftUpperArm': 'LeftShoulder', 'LeftLowerArm': 'LeftUpperArm', 'LeftHand': 'LeftLowerArm',
    'RightShoulder': 'Chest', 'RightUpperArm': 'RightShoulder', 'RightLowerArm': 'RightUpperArm', 'RightHand': 'RightLowerArm',
    'LeftUpperLeg': 'Hips', 'LeftLowerLeg': 'LeftUpperLeg', 'LeftFoot': 'LeftLowerLeg', 'LeftToes': 'LeftFoot',
    'RightUpperLeg': 'Hips', 'RightLowerLeg': 'RightUpperLeg', 'RightFoot': 'RightLowerLeg', 'RightToes': 'RightFoot'
}
BONE_NAMES = sorted(list(PARENTS.keys()))
BONE_MAP = {name: i for i, name in enumerate(BONE_NAMES)}
BONE_RADII = {b: 0.05 for b in BONE_NAMES} # 뼈대 Capsulize

# 1. 물리 엔진 핵심 모듈
def apply_auto_calibration(tensor_data):
    frame0 = tensor_data[0].numpy()
    idx_hips, idx_neck = BONE_MAP['Hips'], BONE_MAP['Neck']
    idx_rupper, idx_lupper = BONE_MAP['RightUpperLeg'], BONE_MAP['LeftUpperLeg']
    
    up = frame0[idx_neck, :3] - frame0[idx_hips, :3]
    up /= np.linalg.norm(up) #Up 정의
    right = frame0[idx_rupper, :3] - frame0[idx_lupper, :3]
    right /= np.linalg.norm(right) #Right 정의
    forward = np.cross(up, right) 
    
    rot_matrix = R.from_matrix(np.column_stack((right, forward, up)).T) 
    
    for f in range(tensor_data.shape[0]):
        pos = tensor_data[f, :, :3].numpy()
        tensor_data[f, :, :3] = torch.tensor(rot_matrix.apply(pos), dtype=torch.float32)
        quat = tensor_data[f, :, 3:].numpy()
        new_rot = rot_matrix * R.from_quat(quat[:, [0,1,2,3]])
        tensor_data[f, :, 3:] = torch.tensor(new_rot.as_quat(), dtype=torch.float32)
    return tensor_data

def apply_hips_centering(tensor_data):
    """
    tensor_data: [Frames, Joints, 7] (Pos 3, Quat 4)
    모든 프레임의 모든 관절 위치(Pos)에서 Hips의 위치를 뺌
    """
    # Hips index
    idx_hips = BONE_MAP['Hips']
    
    # 1. Hips의 위치 데이터만 추출 [Frames, 3]
    hips_pos = tensor_data[:, idx_hips, :3].clone()
    
    # 2. 모든 프레임의 모든 관절 위치에서 Hips 위치를 뺍니다.
    # tensor_data[:, :, :3] = [Frames, Joints, 3] 
    # hips_pos.unsqueeze(1) = [Frames, 1, 3]
    tensor_data[:, :, :3] = tensor_data[:, :, :3] - hips_pos.unsqueeze(1)
    
    return tensor_data

def parse_csv_to_tensor(file_path):
    df = pd.read_csv(file_path)
    frames = sorted(df['Frame'].unique())
    motion_list = []
    
    for f in frames:
        frame_data = df[df['Frame'] == f].set_index('BoneName')
        global_pos, global_rot = {b: np.zeros(3) for b in BONE_NAMES}, {b: R.identity() for b in BONE_NAMES}
        
        for bone in PARENTS.keys():
            if bone not in frame_data.index: continue
            row = frame_data.loc[bone]
            local_p = np.array([row['px'], row['pz'], row['py']])
            lr = R.from_quat([-row['qx'], -row['qz'], -row['qy'], row['qw']])
            if bone != 'Hips': local_p *= 100.0
            p_bone = PARENTS[bone]
            if p_bone is None or p_bone not in global_pos:
                global_pos[bone], global_rot[bone] = local_p, lr
            else:
                global_pos[bone] = global_pos[p_bone] + global_rot[p_bone].apply(local_p)
                global_rot[bone] = global_rot[p_bone] * lr
        
        frame_tensor = torch.zeros((len(BONE_NAMES), 7))
        for b, idx in BONE_MAP.items():
            frame_tensor[idx] = torch.cat([torch.tensor(global_pos[b]), torch.tensor(global_rot[b].as_quat())])
        motion_list.append(frame_tensor)
    return torch.stack(motion_list)

# 2. 데이터셋 및 파이프라인
# 30 Frame 미만인 파일은 사용 X
class BandaiMotionDataset(Dataset):
    def __init__(self, processed_dir, seq_len=30):
        self.seq_len = seq_len
        raw_file_list = glob.glob(os.path.join(processed_dir, "*.pt"))
        if not raw_file_list: 
            raise ValueError("데이터셋이 비어있습니다. PREPROCESS 모드를 먼저 실행하세요.")
        
        self.file_list = []
        print("🔍 데이터셋 무결성 검사 중...")
        
        # 30프레임 이상인 데이터만 골라내어 학습 목록에 추가합니다.
        for f in raw_file_list:
            motion_shape = torch.load(f).shape[0]
            if motion_shape >= self.seq_len:
                self.file_list.append(f)
                
        if not self.file_list:
            raise ValueError(f"모든 데이터가 {self.seq_len} 프레임보다 짧습니다.")
            
        print(f"✅ 총 {len(raw_file_list)}개 중 {len(self.file_list)}개의 유효한 데이터를 로드했습니다.")
        
    def __len__(self): 
        return len(self.file_list)
        
    def __getitem__(self, idx):
        motion = torch.load(self.file_list[idx])
        max_start = motion.shape[0] - self.seq_len
        start = np.random.randint(0, max_start) if max_start > 0 else 0
        return motion[start : start + self.seq_len]

# 3. main
if __name__ == "__main__":
    # MODE: 전역 정의
    if MODE == "PREPROCESS":
        os.makedirs(PT_DIR, exist_ok=True)
        all_csv = glob.glob(os.path.join(CSV_DIR, "**", "*.csv"), recursive=True)
        for f in all_csv:
            save_path = os.path.join(PT_DIR, os.path.basename(f).replace('.csv', '.pt'))
            if not os.path.exists(save_path):
                print(f"Processing: {f}")
                raw = parse_csv_to_tensor(f)
                
                # 순서: 1. 방향 정렬 -> 2. 위치 정규화
                calibrated = apply_auto_calibration(raw)
                final_data = apply_hips_centering(calibrated)
                
                torch.save(final_data, save_path)
        print("✅ 전처리 완료.")

    elif MODE == "VISUALIZE":
        """
        출력 전체 확인
        pt_files = glob.glob(os.path.join(PT_DIR, "*.pt"))
        motion = torch.load(pt_files[0]) # 전체 128프레임이 그대로 로드됨
        print(f"🎬 전체 데이터 로드 완료: {motion.shape}")
        """
        
        VISUALIZE_TYPE = "COMPARE" #SINGLE: 변환한 단일 .pth 파일 / COMPARE: demo_maker / inference로 변환한 비교 파일
        
        if VISUALIZE_TYPE == "SINGLE":
            TARGET_FILE = r"processed_motions/dataset-2_walk-turn-right_exhausted_027.pt" 

            # 1. 데이터 로드 확인
            if TARGET_FILE and os.path.exists(TARGET_FILE):
                print(f"지정된 특정 파일 로드 중: {TARGET_FILE}")
                motion = torch.load(TARGET_FILE)
                # GPU에 올라가 있는 텐서일 경우를 대비해 CPU로 내림
                if motion.device.type != 'cpu':
                    motion = motion.cpu()
            else:
                print("지정된 파일이 없거나 경로가 잘못되어, 데이터셋에서 무작위로 로드합니다.")
                dataset = BandaiMotionDataset(PT_DIR, seq_len=100)
                if len(dataset) == 0:
                    print("데이터가 없습니다. PREPROCESS 모드를 먼저 실행하세요.")
                    import sys; sys.exit()
                motion = dataset[0] # [Frames, Joints, 7]

            print(f"🎬 데이터 로드 완료: {motion.shape} (Frames, Joints, 7)")

            # 2. 렌더링 설정
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Single Motion Viewer", pad=10)
            
            # 뼈대 연결 정보 생성
            bones_to_draw = [(PARENTS[b], b) for b in BONE_NAMES if PARENTS[b] in BONE_NAMES]
            lines = [ax.plot([], [], [], 'o-', lw=2, color='blue')[0] for _ in bones_to_draw]

            ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(0, 1.8)
            ax.view_init(elev=15, azim=45)

            def update(frame_idx):
                # motion[frame_idx]는 [Joints, 7] 형태
                frame_pos = motion[frame_idx, :, :3] # [Joints, 3]
                
                for i, (parent_name, child_name) in enumerate(bones_to_draw):
                    # BONE_MAP을 사용하여 해당 뼈대의 인덱스 추출
                    p_idx = BONE_MAP[parent_name]
                    c_idx = BONE_MAP[child_name]
                    
                    p_pos = frame_pos[p_idx]
                    c_pos = frame_pos[c_idx]
                    
                    lines[i].set_data([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]])
                    lines[i].set_3d_properties([p_pos[2], c_pos[2]])
                
                # 3D 축 범위 설정 (데이터가 보일 수 있도록)
                ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 2)
                return lines

            ani = animation.FuncAnimation(fig, update, frames=motion.shape[0], interval=33, blit=False)
            
            def on_scroll(event):
                scale_factor = 0.9 if event.button == 'up' else 1.1
                ax.set_xlim([x * scale_factor for x in ax.get_xlim()])
                ax.set_ylim([y * scale_factor for y in ax.get_ylim()])
                ax.set_zlim([z * scale_factor for z in ax.get_zlim()])
                fig.canvas.draw_idle()
                fig.canvas.mpl_connect('scroll_event', on_scroll)
            plt.show()
                  
        elif VISUALIZE_TYPE == "COMPARE":
            print("Before & After 비교 시각화 모드입니다.")
            # demo_maker -> demo_results
            # inference -> inference_results
            
            from physics_module import DifferentiablePhysics
            DEVICE = 'cpu'
            physics_engine = DifferentiablePhysics(PARENTS, {b: 0.05 for b in BONE_NAMES}).to(DEVICE)

            orig_path = "demo_results/sample_original.pt"
            corr_path = "demo_results/sample_corrected.pt"

            if not os.path.exists(orig_path) or not os.path.exists(corr_path):
                print("결과를 찾을 수 없습니다. 먼저 python inference/demo_maker.py를 실행하세요.")
                import sys; sys.exit()
                
            motion_orig = torch.load(orig_path).cpu()
            motion_corr = torch.load(corr_path).cpu()
            
            # 시각화를 위해 캡슐 뼈대 그룹 재정의 (어깨 제외)
            CAPSULE_BONES = {
                'Hips_Chest': ['Spine', 'Chest', 'Neck', 'Head'],
                'LeftArm': ['LeftLowerArm', 'LeftHand'],
                'RightArm': ['RightLowerArm', 'RightHand'],
                'LeftLeg': ['LeftLowerLeg', 'LeftFoot', 'LeftToes'],
                'RightLeg': ['RightLowerLeg', 'RightFoot', 'RightToes']
            }
            
            # [핵심] 어깨(Shoulder)를 제외한 안전한 페어
            TEST_PAIRS = [
                ('Hips_Chest', 'LeftArm', 'Hips', 'Chest', 'LeftLowerArm', 'LeftHand'),
                ('Hips_Chest', 'RightArm', 'Hips', 'Chest', 'RightLowerArm', 'RightHand'),
                ('LeftArm', 'RightArm', 'LeftLowerArm', 'LeftHand', 'RightLowerArm', 'RightHand'),
                ('LeftLeg', 'RightLeg', 'LeftLowerLeg', 'LeftFoot', 'RightLowerLeg', 'RightFoot')
            ]

            fig = plt.figure(figsize=(14, 7))
            fig.suptitle("AI Motion Correction: Before & After (Physics Fixed)", fontsize=16, fontweight='bold')

            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_title("Before (Original)", pad=10)
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_title("After (AI Corrected)", pad=10)

            bones_to_draw = [(PARENTS[b], b) for b in BONE_NAMES if PARENTS[b] in BONE_NAMES]

            lines_orig_bone = [ax1.plot([], [], [], 'o-', lw=2.0, markersize=3, color='salmon')[0] for _ in bones_to_draw]
            lines_orig_capsule = [ax1.plot([], [], [], '-', lw=18, color='salmon', alpha=0.15)[0] for _ in bones_to_draw]
            lines_corr_bone = [ax2.plot([], [], [], 'o-', lw=2.0, markersize=3, color='dodgerblue')[0] for _ in bones_to_draw]
            lines_corr_capsule = [ax2.plot([], [], [], '-', lw=18, color='dodgerblue', alpha=0.15)[0] for _ in bones_to_draw]

            for ax in [ax1, ax2]:
                ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(0, 1.8)
                ax.view_init(elev=15, azim=45)

            def update_compare(frame_idx):
                pos_orig_t = motion_orig[frame_idx, :, :3]
                pos_corr_t = motion_corr[frame_idx, :, :3]

                pos_orig_np = pos_orig_t.numpy()
                pos_corr_np = pos_corr_t.numpy()

                red_bones_orig, red_bones_corr = set(), set()

                for cap1, cap2, p1, c1, p2, c2 in TEST_PAIRS:
                    dist_orig = physics_engine.capsule_distance(
                        pos_orig_t[BONE_MAP[p1]], pos_orig_t[BONE_MAP[c1]], 
                        pos_orig_t[BONE_MAP[p2]], pos_orig_t[BONE_MAP[c2]]
                    ).item() 
                    if dist_orig < 0.1:
                        red_bones_orig.update(CAPSULE_BONES[cap1]); red_bones_orig.update(CAPSULE_BONES[cap2])
                    
                    dist_corr = physics_engine.capsule_distance(
                        pos_corr_t[BONE_MAP[p1]], pos_corr_t[BONE_MAP[c1]], 
                        pos_corr_t[BONE_MAP[p2]], pos_corr_t[BONE_MAP[c2]]
                    ).item()
                    if dist_corr < 0.1: 
                        red_bones_corr.update(CAPSULE_BONES[cap1]); red_bones_corr.update(CAPSULE_BONES[cap2])

                for i, (parent_name, child_name) in enumerate(bones_to_draw):
                    p_idx, c_idx = BONE_MAP[parent_name], BONE_MAP[child_name]

                    # 왼쪽
                    po_p, po_c = pos_orig_np[p_idx], pos_orig_np[c_idx]
                    orig_color = 'red' if child_name in red_bones_orig else 'dimgray'
                    lines_orig_bone[i].set_data([po_p[0], po_c[0]], [po_p[1], po_c[1]])
                    lines_orig_bone[i].set_3d_properties([po_p[2], po_c[2]])
                    lines_orig_bone[i].set_color(orig_color)
                    lines_orig_capsule[i].set_data([po_p[0], po_c[0]], [po_p[1], po_c[1]])
                    lines_orig_capsule[i].set_3d_properties([po_p[2], po_c[2]])
                    lines_orig_capsule[i].set_color('red' if child_name in red_bones_orig else 'salmon')
                    lines_orig_capsule[i].set_alpha(0.4 if child_name in red_bones_orig else 0.15)

                    # 오른쪽
                    pc_p, pc_c = pos_corr_np[p_idx], pos_corr_np[c_idx]
                    corr_color = 'red' if child_name in red_bones_corr else 'dodgerblue'
                    lines_corr_bone[i].set_data([pc_p[0], pc_c[0]], [pc_p[1], pc_c[1]])
                    lines_corr_bone[i].set_3d_properties([pc_p[2], pc_c[2]])
                    lines_corr_bone[i].set_color(corr_color)
                    lines_corr_capsule[i].set_data([pc_p[0], pc_c[0]], [pc_p[1], pc_c[1]])
                    lines_corr_capsule[i].set_3d_properties([pc_p[2], pc_c[2]])
                    lines_corr_capsule[i].set_color('red' if child_name in red_bones_corr else 'dodgerblue')
                    lines_corr_capsule[i].set_alpha(0.4 if child_name in red_bones_corr else 0.15)

                return lines_orig_bone + lines_orig_capsule + lines_corr_bone + lines_corr_capsule

            ani = animation.FuncAnimation(fig, update_compare, frames=motion_orig.shape[0], interval=50, blit=False)

            def on_mouse_move(event):
                if event.inaxes == ax1:
                    ax2.view_init(elev=ax1.elev, azim=ax1.azim)
                    fig.canvas.draw_idle()
                elif event.inaxes == ax2:
                    ax1.view_init(elev=ax2.elev, azim=ax2.azim)
                    fig.canvas.draw_idle()
            fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

            from matplotlib.widgets import Button
            plt.subplots_adjust(bottom=0.15)
            ax_play = plt.axes([0.45, 0.05, 0.1, 0.05])
            btn_play = Button(ax_play, 'Pause', hovercolor='0.9')
            is_playing = [True]

            def toggle_play(event=None):
                if is_playing[0]:
                    ani.pause()
                    btn_play.label.set_text('Play (Space)')
                else:
                    ani.resume()
                    btn_play.label.set_text('Pause (Space)')
                is_playing[0] = not is_playing[0]
                fig.canvas.draw_idle()
            btn_play.on_clicked(toggle_play)

            def on_key_press(event):
                if event.key == ' ':
                    toggle_play()
            fig.canvas.mpl_connect('key_press_event', on_key_press)
            
            def on_scroll(event):
                scale_factor = 0.9 if event.button == 'up' else 1.1
                for ax in [ax1, ax2]:
                    ax.set_xlim([x * scale_factor for x in ax.get_xlim()])
                    ax.set_ylim([y * scale_factor for y in ax.get_ylim()])
                    ax.set_zlim([z * scale_factor for z in ax.get_zlim()])
                fig.canvas.draw_idle()
            fig.canvas.mpl_connect('scroll_event', on_scroll)
            plt.show()