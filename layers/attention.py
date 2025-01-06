import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms.v2.functional import pad_mask
import math


config = dict(
    max_position_embeddings = 2048,
    num_heads = 12,
    num_layers = 12,
    vocab_size = 50257,
    embedding_size = 768,
)
class GPTNeoModel(nn.Module):
    def __init__(self):
        super(GPTNeoModel, self).__init__()
        self.embd = Embedding()
        self.block = nn.Sequential(
            GPTNeoBlock(),
            GPTNeoBlock(),
            GPTNeoBlock(),
            GPTNeoBlock(),
            GPTNeoBlock(),
            GPTNeoBlock(),
            GPTNeoBlock(),
            GPTNeoBlock(),
            GPTNeoBlock(),
            GPTNeoBlock(),
            GPTNeoBlock(),
            GPTNeoBlock()
        )
        self.final_ln = nn.LayerNorm(768)
        self.final_linear = nn.Linear(768, config['vocab_size'])
    def forward(self, x):
        x = self.embd(x)
        x = self.block(x)
        x = self.final_ln(x)
        return self.final_linear(x)

    @torch.no_grad()
    def generate(self, input_ids, max_length=50, top_k=50, top_p=0.9, temperature=1.0):

        generated_ids = input_ids
        for _ in range(max_length):
            # 모델 출력
            logits = self.forward(generated_ids)

            logits = logits[:, -1, :]


            logits = logits / temperature

            # Top-k 샘플링
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits[logits < top_k_values[:, -1].unsqueeze(-1)] = -float("Inf")

            # Top-p 샘플링
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                for i, indices in enumerate(sorted_indices):
                    logits[i, indices[sorted_indices_to_remove[i]]] = -float("Inf")

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        return generated_ids


class GPTNeoBlock(nn.Module):
    def __init__(self):
        super(GPTNeoBlock, self).__init__()
        self.ln = nn.LayerNorm(768)
        self.attn = GPTNeoSelfAttention(embed_dim=768, num_heads=12)# 마지막 레이어에 이미 노말라이즈 함
        self.head = GPTNeoMLP()
    def forward(self, x):
        x = self.ln(x)
        x = self.attn(x)
        x = self.head(x)
        return x
'''
load
'''
class GPTNeoMLP(nn.Module):
    def __init__(self):
        super(GPTNeoMLP, self).__init__()
        self.linear1 = nn.Linear(768, 3072)
        self.linear2 = nn.Linear(3072, 768)
        self.ln = nn.GELU()
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        x = self.ln(self.linear1(x))
        x = self.ln(self.linear2(x))
        x = self.drop(x)
        return x


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_size):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.embedding_size = embedding_size

        position = torch.arange(0, max_position_embeddings).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2) * -(math.log(10000.0) / embedding_size)
        )
        pe = torch.zeros(max_position_embeddings, embedding_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        return self.pe[:seq_len, :].unsqueeze(0).expand(input_ids.size(0), -1, -1)

'''
load
'''
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.word_embedding = nn.Embedding(config["vocab_size"], config["embedding_size"])
        self.position_embedding = SinusoidalPositionEmbedding(config["max_position_embeddings"], config["embedding_size"])
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        word_embeddings = self.word_embedding(input_ids)  # (B, S, C)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)  # (S,)
        position_embeddings = self.position_embedding(position_ids.unsqueeze(0).expand_as(input_ids))  # (B, S, C)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


'''
load
'''
class GPTNeoSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, attn_drop=0.):
        super(GPTNeoSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ln = nn.LayerNorm(embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.embed_dim = embed_dim

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape


        logical_mask = self.logical_mask_generator(N).to(x.device)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        head_dim = C // 12  # 각 헤드의 차원
        q = q.view(B, N, 12, head_dim).transpose(1, 2)  # (B, N, C) -> (B, 12, N, head_dim)
        k = k.view(B, N, 12, head_dim).transpose(1, 2)  # (B, N, C) -> (B, 12, N, head_dim)
        v = v.view(B, N, 12, head_dim).transpose(1, 2)  # (B, N, C) -> (B, 12, N, head_dim)

        energy = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        energy = energy + logical_mask
        attention = torch.softmax(energy, dim=-1)
        attention = self.attn_drop(attention)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).reshape(B, N, C)
        output = self.out_proj(output)
        return self.ln(output + x)

    def logical_mask_generator(self,seq_len, fill_value=-1e4):

        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float32))
        mask = mask.masked_fill(mask == 0, fill_value)

        return mask.unsqueeze(0).unsqueeze(0) # 1,1,s,s

    def padding_mask_generator(self, x):
        pad_mask = torch.zeros(size = (1,1,config["max_position_embeddings"],1), dtype=torch.float32)
        S = x.shape[1]
        pad_mask[:, :, :S, :] = 1.0
        return pad_mask

if __name__ == "__main__":
    attn = GPTNeoSelfAttention(768)
    print(f'attention layer test : {attn.forward(torch.randn(size=(4,2048,768))).shape} ')

    test_embd  = torch.randint(low = 0, high = config["max_position_embeddings"], size=(4,2048))
    print(test_embd.shape)
    module = Embedding()
    print(f'embedding test :{module.forward(test_embd).shape} ')

    model = GPTNeoModel()
    print(model)
    print((model(test_embd)).shape)
    from torchinfo import summary
    summary(
        model,
        input_data=test_embd,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=1000000,
    )

