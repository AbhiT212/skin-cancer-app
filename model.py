import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvNextTransformerStream(nn.Module):
    def __init__(self, transformer_dim=512, num_heads=8):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        convnext = models.convnext_tiny(weights=None)
        self.backbone = convnext.features
        self.projection = nn.Conv2d(768, transformer_dim, kernel_size=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, 400, transformer_dim)) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.projection(features)
        b, c, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        
        if tokens.shape[1] != self.pos_embedding.shape[1]:
            pos_emb = F.interpolate(self.pos_embedding.transpose(1,2), size=tokens.shape[1], mode='linear').transpose(1,2)
        else:
            pos_emb = self.pos_embedding
            
        tokens = tokens + pos_emb
        trans_out = self.transformer(tokens)
        out = trans_out.transpose(1, 2)
        return self.pool(out).flatten(1)

class EfficientNetStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.effnet = models.efficientnet_b0(weights=None)
        self.backbone = self.effnet.features
        self.pool = self.effnet.avgpool
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return x.flatten(1)

class SkipAttentionFusion(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_q // num_heads
        self.scale = self.dim_head ** -0.5
        self.to_q = nn.Linear(dim_q, dim_q, bias=False)
        self.to_k = nn.Linear(dim_kv, dim_q, bias=False)
        self.to_v = nn.Linear(dim_kv, dim_q, bias=False)
        self.proj = nn.Linear(dim_q, dim_q)
        self.norm = nn.LayerNorm(dim_q)
        
    def forward(self, x_q, x_kv):
        B, _ = x_q.shape
        q = self.to_q(x_q).reshape(B, 1, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        k = self.to_k(x_kv).reshape(B, 1, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        v = self.to_v(x_kv).reshape(B, 1, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, -1)
        out = self.proj(out)
        return self.norm(x_q + out)

class SkinCancerHybrid_Pro(nn.Module):
    def __init__(self, num_classes=7, num_meta_features=19):
        super().__init__()
        self.stream_a = ConvNextTransformerStream(transformer_dim=512)
        self.stream_b = EfficientNetStream() 
        self.skip_attn = SkipAttentionFusion(dim_q=512, dim_kv=1280)
        
        self.meta_net = nn.Sequential(
            nn.Linear(num_meta_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, meta):
        feat_global = self.stream_a(x)
        feat_local = self.stream_b(x)
        fused_img = self.skip_attn(feat_global, feat_local)
        feat_meta = self.meta_net(meta)
        combined = torch.cat((fused_img, feat_meta), dim=1)
        return self.classifier(combined)