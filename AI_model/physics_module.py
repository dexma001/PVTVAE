import torch
import torch.nn as nn

class DifferentiablePhysics(nn.Module):
    def __init__(self, parents, bone_radii):
        super().__init__()
        self.parents = parents
        self.bone_radii = {k: torch.tensor(v, dtype=torch.float32) for k, v in bone_radii.items()}

    def quat_multiply(self, q1, q2):
        # q: [..., 4] (x, y, z, w)
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([x, y, z, w], dim=-1)

    def capsule_distance(self, p1, q1, p2, q2):
            """
            미분 가능한 3D 선분 간 최단 거리 알고리즘 (Lumelsky's Algorithm Tensorized)
            """
            SMALL_NUM = 1e-8
            
            # 방향 벡터 계산
            u = q1 - p1
            v = q2 - p2
            w = p1 - p2

            a = (u * u).sum(-1)
            b = (u * v).sum(-1)
            c = (v * v).sum(-1)
            d = (u * w).sum(-1)
            e = (v * w).sum(-1)
            D = a * c - b * b

            # 파라미터 s와 t의 분모(Denominator) 초기화
            sD = D
            tD = D

            # s와 t의 분자(Numerator) 계산 (평행선 예외 처리 포함)
            sN = torch.where(D < SMALL_NUM, torch.zeros_like(D), b * e - c * d)
            tN = torch.where(D < SMALL_NUM, e, a * e - b * d)
            tD = torch.where(D < SMALL_NUM, c, tD)

            # 1. s 파라미터를 [0, 1] 구간으로 클램핑(Clamping)
            s_less_0 = sN < 0.0
            sN = torch.where(s_less_0, torch.zeros_like(sN), sN)
            tN = torch.where(s_less_0, e, tN)
            tD = torch.where(s_less_0, c, tD)

            s_greater_d = sN > sD
            sN = torch.where(s_greater_d, sD, sN)
            tN = torch.where(s_greater_d, e + b, tN)
            tD = torch.where(s_greater_d, c, tD)

            # 2. t 파라미터를 [0, 1] 구간으로 클램핑
            zeros = torch.zeros_like(a) # 0.0을 a와 같은 형태의 GPU 텐서로 생성 (추가)

            t_less_0 = tN < 0.0
            tN = torch.where(t_less_0, torch.zeros_like(tN), tN)
            sN_new_t0 = torch.clamp(-d, min=zeros, max=a) # 텐서끼리 매칭 (수정)
            sN = torch.where(t_less_0, sN_new_t0, sN)
            sD = torch.where(t_less_0, a, sD)

            t_greater_d = tN > tD
            tN = torch.where(t_greater_d, tD, tN)
            sN_new_t1 = torch.clamp(-d + b, min=zeros, max=a) # 텐서끼리 매칭 (수정)
            sN = torch.where(t_greater_d, sN_new_t1, sN)
            sD = torch.where(t_greater_d, a, sD)
            
            # 3. 최종 파라미터 sc, tc 계산 (0으로 나누기 방지 - 안전한 나눗셈 적용)
            safe_sD = torch.clamp(sD, min=SMALL_NUM)
            safe_tD = torch.clamp(tD, min=SMALL_NUM)
            
            sc = torch.where(torch.abs(sN) < SMALL_NUM, torch.zeros_like(sN), sN / safe_sD)
            tc = torch.where(torch.abs(tN) < SMALL_NUM, torch.zeros_like(tN), tN / safe_tD)

            # 4. 차원 맞추기 및 최단 거리 벡터 계산
            sc = sc.unsqueeze(-1)
            tc = tc.unsqueeze(-1)
            
            dP = w + (sc * u) - (tc * v)
            
            # 유클리드 거리 반환 (루트 0 미분 폭발 방지용 + 1e-8 추가)
            return torch.sqrt(torch.sum(dP * dP, dim=-1) + 1e-8)

    def get_collision_loss(self, global_pos, colliding_pairs):
        loss = 0.0
        for (p1, c1), (p2, c2) in colliding_pairs:
            dist = self.capsule_distance(global_pos[p1], global_pos[c1], global_pos[p2], global_pos[c2])
            threshold = self.bone_radii[c1] + self.bone_radii[c2]
            # 충돌 깊이만큼 loss 추가 (충돌하지 않으면 0)
            loss += torch.relu(threshold - dist).mean()
        return loss
    
    def get_collision_loss_from_tensor(self, tensor_data, colliding_pairs):
        """
        AI 모델이 출력한 [Batch, Seq, Joints, 7] 텐서를 입력받아
        물리 엔진이 계산할 수 있도록 관절별 위치(Pos) 딕셔너리로 분리.
        """
        # 1. 뼈대 이름 순서대로 인덱스 맵 생성 (dataset_pipeline과 동일한 정렬)
        bone_names = sorted(list(self.parents.keys()))
        bone_map = {name: i for i, name in enumerate(bone_names)}
        
        global_pos = {}
        
        # 2. 전체 배치와 시퀀스에 대해 관절별 위치값(x, y, z)만 추출
        for bone_name, idx in bone_map.items():
            # tensor_data[..., idx, :3] -> 마지막 차원의 0~2번 인덱스(Pos)만 가져옴
            global_pos[bone_name] = tensor_data[..., idx, :3]
            
        # 3. 기존에 작성한 충돌 계산 로직으로 넘김 (배치 연산 자동 지원)
        return self.get_collision_loss(global_pos, colliding_pairs)