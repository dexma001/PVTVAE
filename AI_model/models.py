import torch
import torch.nn as nn

class PVTVAE(nn.Module):
    def __init__(self, input_dim=147, latent_dim=64, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        # 1. Input Projection: Motion Data -> 고차원 특징 공간
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 1+. Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 30, d_model))
        
        # 2. Encoder Head
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Latent Head: 평균(mu)과 로그 분산(logvar) 예측
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_var = nn.Linear(d_model, latent_dim)
        
        # 4. Decoder Head
        self.decoder_proj = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 5. Output Projection: 다시 원래 모션 차원으로
        self.output_layer = nn.Linear(d_model, input_dim)

    def reparameterize(self, mu, logvar):
        """VAE의 핵심: 샘플링을 통해 잠재 공간의 연속성 확보"""
        if self.training: # 학습 중일 때만 노이즈 추가
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else: # 실시간 추론(평가) 시에는 확정적인 결과(mu)만 출력
            return mu

    def forward(self, x):
        # x: [Batch, Seq, 147]
        x = self.input_proj(x)
        
        # Positional Encoding
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Encoder: 전체 시퀀스 특징 추출
        encoded = self.encoder_transformer(x)
        
        # 마지막 프레임의 특징을 대표값으로 사용 (혹은 pooling 사용 가능)
        feat = encoded[:, -1, :] 
        mu, logvar = self.fc_mu(feat), self.fc_var(feat)
        z = self.reparameterize(mu, logvar)
        
        # Decoder: z를 시퀀스 길이만큼 확장 후 디코딩
        # z: [Batch, latent_dim] -> [Batch, Seq, d_model]
        z_expanded = self.decoder_proj(z).unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Positional Encoding in Decoder layer
        z_expanded = z_expanded + self.pos_embedding[:, :x.size(1), :]
        
        # Decoder Transformer
        decoded = self.decoder_transformer(z_expanded, encoded) 
        
        out = self.output_layer(decoded)
        return out, mu, logvar
    
if __name__ == "__main__":
    # 모델 테스트
    model = PVTVAE(input_dim=147, latent_dim=64)
    # Batch=8, Seq=30, Dim=147 (우리의 데이터 형태)
    dummy_data = torch.randn(8, 30, 147) 
    
    output, mu, logvar = model(dummy_data)
    print(f"입력 크기: {dummy_data.shape}")
    print(f"출력 크기: {output.shape}") # [8, 30, 147] 이 나와야 합니다.
    print(f"잠재 변수 크기: {mu.shape}") # [8, 64]