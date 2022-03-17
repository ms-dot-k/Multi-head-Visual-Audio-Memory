import torch
from torch import nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self, radius=16.0, n_slot=112, n_head=8, dim=512, diff_aud_vid=False):
        super().__init__()

        self.dav = diff_aud_vid

        self.head = n_head
        self.slot = n_slot

        self.key = nn.Parameter(torch.Tensor(int(n_head * n_slot), int(512 / n_head)), requires_grad=True)
        nn.init.normal_(self.key, 0, 0.5)
        self.value = nn.Parameter(torch.Tensor(n_slot, 512), requires_grad=True)
        nn.init.normal_(self.value, 0, 0.5)

        if self.dav:
            self.linear = nn.Linear(512 * n_head, dim)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.norm3 = nn.LayerNorm(dim)
            self.v_up = nn.Linear(512, dim)
        else:
            self.linear = nn.Linear(512 * n_head, 512)
            self.norm1 = nn.LayerNorm(512)
            self.norm2 = nn.LayerNorm(512)
            self.norm3 = nn.LayerNorm(512)

        self.q_embd = nn.Linear(dim, 512)
        self.v_embd = nn.Linear(512, 512)

        self.dropout = nn.Dropout(0.5)

        self.radius = radius
        self.softmax1 = nn.Softmax(2)
        self.softmax2 = nn.Softmax(1)

    def forward(self, query, value=None, inference=False, cha_first=False):
        if cha_first:
            query = query.transpose(1, 2).contiguous()
        # B, S, 512
        B, S, C = query.size()
        mer_query = query.view(B * S, -1)
        tr_fusion, recon_loss, contrastive_loss = None, torch.zeros(1), torch.zeros(1)

        key_norm = F.normalize(self.key.view(self.head, self.slot, -1), dim=2) #n_head, n_slot, head_dim
        embd_query = self.q_embd(mer_query) #B*S, n_head * head_dim
        embd_query = embd_query.view(B * S, self.head, -1)  #BS, n_head, head_dim
        embd_query = F.normalize(embd_query, dim=2)

        key_sim = torch.einsum('bhd,hsd->bhs', embd_query, key_norm)    #BS, n_head, n_slot
        key_add = self.softmax1(self.radius * key_sim)  # BS, n_head, n_slot

        m_head_aud = torch.matmul(key_add, self.value.detach()) # BS, n_head, 512
        m_head_aud = m_head_aud.view(B * S, -1)     #BS, n_head*512
        vir_aud = self.norm2(self.linear(m_head_aud))   #BS, 512

        te_fusion = self.norm1(query + vir_aud.view(B, S, -1))
        te_fusion = self.dropout(te_fusion)

        # Update
        if not inference:
            mer_value = value.view(B * S, -1)   #BS,512
            embd_value = self.v_embd(mer_value.detach())
            value_norm = F.normalize(self.value, dim=1) #n_slot,512
            value_sim = F.linear(F.normalize(embd_value, dim=1), value_norm) #BS, n_slot
            value_add = self.softmax2(self.radius * value_sim)

            aud = torch.matmul(value_add, self.value)   #BS,512

            contrastive_loss = 0.5 * torch.abs(torch.eye(self.slot).cuda() - torch.matmul(value_norm, value_norm.transpose(0, 1))).sum()    #n_slot,n_slot
            contrastive_loss = contrastive_loss.unsqueeze(0)

            recon_loss = torch.abs(1.0 - F.cosine_similarity(aud, mer_value.detach(), 1))  #BS
            recon_loss = recon_loss.view(B, S).sum(1)   #B

            if self.dav:
                aud = self.v_up(aud)
            aud = self.norm3(aud)
            tr_fusion = self.norm1(query + aud.view(B, S, -1))
            tr_fusion = self.dropout(tr_fusion)

        if cha_first:
            te_fusion = te_fusion.transpose(1, 2).contiguous()
            if tr_fusion is not None:
                tr_fusion = tr_fusion.transpose(1, 2).contiguous()
        return te_fusion, tr_fusion, recon_loss, contrastive_loss
