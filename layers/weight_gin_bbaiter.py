from layers.attention import GPTNeoModel, SinusoidalPositionEmbedding
import torch
from torchinfo import summary
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

miunsi = GPTNeoModel()

tmp = model.state_dict()

position_embedding = SinusoidalPositionEmbedding(max_position_embeddings=2048, embedding_size=768)

loader_state = dict()

miunsi_state = miunsi.state_dict()


for i in miunsi_state.keys():
    print(f'{i}\n')

def gin_bbai():

    #block1
    miunsi_state['embd.word_embedding.weight'] = tmp['transformer.wte.weight']
    miunsi_state['block.0.ln.weight'] = tmp['transformer.h.0.ln_1.weight']
    miunsi_state['block.0.attn.ln.weight'] = tmp['transformer.h.0.ln_1.weight']
    miunsi_state['block.0.attn.k_proj.weight'] = tmp['transformer.h.0.attn.attention.k_proj.weight']
    miunsi_state['block.0.attn.v_proj.weight'] = tmp['transformer.h.0.attn.attention.v_proj.weight']
    miunsi_state['block.0.attn.q_proj.weight'] = tmp['transformer.h.0.attn.attention.q_proj.weight']
    miunsi_state['block.0.attn.out_proj.weight'] = tmp['transformer.h.0.attn.attention.out_proj.weight']
    miunsi_state['block.0.head.linear1.weight'] = tmp['transformer.h.0.mlp.c_fc.weight']
    miunsi_state['block.0.head.linear2.weight'] = tmp['transformer.h.0.mlp.c_proj.weight']

    # block1
    miunsi_state['block.1.ln.weight'] = tmp['transformer.h.1.ln_1.weight']
    miunsi_state['block.1.attn.ln.weight'] = tmp['transformer.h.1.ln_1.weight']
    miunsi_state['block.1.attn.k_proj.weight'] = tmp['transformer.h.1.attn.attention.k_proj.weight']
    miunsi_state['block.1.attn.v_proj.weight'] = tmp['transformer.h.1.attn.attention.v_proj.weight']
    miunsi_state['block.1.attn.q_proj.weight'] = tmp['transformer.h.1.attn.attention.q_proj.weight']
    miunsi_state['block.1.attn.out_proj.weight'] = tmp['transformer.h.1.attn.attention.out_proj.weight']
    miunsi_state['block.1.head.linear1.weight'] = tmp['transformer.h.1.mlp.c_fc.weight']
    miunsi_state['block.1.head.linear2.weight'] = tmp['transformer.h.1.mlp.c_proj.weight']

    # block2
    miunsi_state['block.2.ln.weight'] = tmp['transformer.h.2.ln_1.weight']
    miunsi_state['block.2.attn.ln.weight'] = tmp['transformer.h.2.ln_1.weight']
    miunsi_state['block.2.attn.k_proj.weight'] = tmp['transformer.h.2.attn.attention.k_proj.weight']
    miunsi_state['block.2.attn.v_proj.weight'] = tmp['transformer.h.2.attn.attention.v_proj.weight']
    miunsi_state['block.2.attn.q_proj.weight'] = tmp['transformer.h.2.attn.attention.q_proj.weight']
    miunsi_state['block.2.attn.out_proj.weight'] = tmp['transformer.h.2.attn.attention.out_proj.weight']
    miunsi_state['block.2.head.linear1.weight'] = tmp['transformer.h.2.mlp.c_fc.weight']
    miunsi_state['block.2.head.linear2.weight'] = tmp['transformer.h.2.mlp.c_proj.weight']

    # block3
    miunsi_state['block.3.ln.weight'] = tmp['transformer.h.3.ln_1.weight']
    miunsi_state['block.3.attn.ln.weight'] = tmp['transformer.h.3.ln_1.weight']
    miunsi_state['block.3.attn.k_proj.weight'] = tmp['transformer.h.3.attn.attention.k_proj.weight']
    miunsi_state['block.3.attn.v_proj.weight'] = tmp['transformer.h.3.attn.attention.v_proj.weight']
    miunsi_state['block.3.attn.q_proj.weight'] = tmp['transformer.h.3.attn.attention.q_proj.weight']
    miunsi_state['block.3.attn.out_proj.weight'] = tmp['transformer.h.3.attn.attention.out_proj.weight']
    miunsi_state['block.3.head.linear1.weight'] = tmp['transformer.h.3.mlp.c_fc.weight']
    miunsi_state['block.3.head.linear2.weight'] = tmp['transformer.h.3.mlp.c_proj.weight']

    # block4
    miunsi_state['block.4.ln.weight'] = tmp['transformer.h.4.ln_1.weight']
    miunsi_state['block.4.attn.ln.weight'] = tmp['transformer.h.4.ln_1.weight']
    miunsi_state['block.4.attn.k_proj.weight'] = tmp['transformer.h.4.attn.attention.k_proj.weight']
    miunsi_state['block.4.attn.v_proj.weight'] = tmp['transformer.h.4.attn.attention.v_proj.weight']
    miunsi_state['block.4.attn.q_proj.weight'] = tmp['transformer.h.4.attn.attention.q_proj.weight']
    miunsi_state['block.4.attn.out_proj.weight'] = tmp['transformer.h.4.attn.attention.out_proj.weight']
    miunsi_state['block.4.head.linear1.weight'] = tmp['transformer.h.4.mlp.c_fc.weight']
    miunsi_state['block.4.head.linear2.weight'] = tmp['transformer.h.4.mlp.c_proj.weight']

    # block5
    miunsi_state['block.5.ln.weight'] = tmp['transformer.h.5.ln_1.weight']
    miunsi_state['block.5.attn.ln.weight'] = tmp['transformer.h.5.ln_1.weight']
    miunsi_state['block.5.attn.k_proj.weight'] = tmp['transformer.h.5.attn.attention.k_proj.weight']
    miunsi_state['block.5.attn.v_proj.weight'] = tmp['transformer.h.5.attn.attention.v_proj.weight']
    miunsi_state['block.5.attn.q_proj.weight'] = tmp['transformer.h.5.attn.attention.q_proj.weight']
    miunsi_state['block.5.attn.out_proj.weight'] = tmp['transformer.h.5.attn.attention.out_proj.weight']
    miunsi_state['block.5.head.linear1.weight'] = tmp['transformer.h.5.mlp.c_fc.weight']
    miunsi_state['block.5.head.linear2.weight'] = tmp['transformer.h.5.mlp.c_proj.weight']

    # block6
    miunsi_state['block.6.ln.weight'] = tmp['transformer.h.6.ln_1.weight']
    miunsi_state['block.6.attn.ln.weight'] = tmp['transformer.h.6.ln_1.weight']
    miunsi_state['block.6.attn.k_proj.weight'] = tmp['transformer.h.6.attn.attention.k_proj.weight']
    miunsi_state['block.6.attn.v_proj.weight'] = tmp['transformer.h.6.attn.attention.v_proj.weight']
    miunsi_state['block.6.attn.q_proj.weight'] = tmp['transformer.h.6.attn.attention.q_proj.weight']
    miunsi_state['block.6.attn.out_proj.weight'] = tmp['transformer.h.6.attn.attention.out_proj.weight']
    miunsi_state['block.6.head.linear1.weight'] = tmp['transformer.h.6.mlp.c_fc.weight']
    miunsi_state['block.6.head.linear2.weight'] = tmp['transformer.h.6.mlp.c_proj.weight']

    # block7
    miunsi_state['block.7.ln.weight'] = tmp['transformer.h.7.ln_1.weight']
    miunsi_state['block.7.attn.ln.weight'] = tmp['transformer.h.7.ln_1.weight']
    miunsi_state['block.7.attn.k_proj.weight'] = tmp['transformer.h.7.attn.attention.k_proj.weight']
    miunsi_state['block.7.attn.v_proj.weight'] = tmp['transformer.h.7.attn.attention.v_proj.weight']
    miunsi_state['block.7.attn.q_proj.weight'] = tmp['transformer.h.7.attn.attention.q_proj.weight']
    miunsi_state['block.7.attn.out_proj.weight'] = tmp['transformer.h.7.attn.attention.out_proj.weight']
    miunsi_state['block.7.head.linear1.weight'] = tmp['transformer.h.7.mlp.c_fc.weight']
    miunsi_state['block.7.head.linear2.weight'] = tmp['transformer.h.8.mlp.c_proj.weight']

    # block8
    miunsi_state['block.8.ln.weight'] = tmp['transformer.h.8.ln_1.weight']
    miunsi_state['block.8.attn.ln.weight'] = tmp['transformer.h.8.ln_1.weight']
    miunsi_state['block.8.attn.k_proj.weight'] = tmp['transformer.h.8.attn.attention.k_proj.weight']
    miunsi_state['block.8.attn.v_proj.weight'] = tmp['transformer.h.8.attn.attention.v_proj.weight']
    miunsi_state['block.8.attn.q_proj.weight'] = tmp['transformer.h.8.attn.attention.q_proj.weight']
    miunsi_state['block.8.attn.out_proj.weight'] = tmp['transformer.h.8.attn.attention.out_proj.weight']
    miunsi_state['block.8.head.linear1.weight'] = tmp['transformer.h.8.mlp.c_fc.weight']
    miunsi_state['block.8.head.linear2.weight'] = tmp['transformer.h.8.mlp.c_proj.weight']

    # block9
    miunsi_state['block.9.ln.weight'] = tmp['transformer.h.9.ln_1.weight']
    miunsi_state['block.9.attn.ln.weight'] = tmp['transformer.h.9.ln_1.weight']
    miunsi_state['block.9.attn.k_proj.weight'] = tmp['transformer.h.9.attn.attention.k_proj.weight']
    miunsi_state['block.9.attn.v_proj.weight'] = tmp['transformer.h.9.attn.attention.v_proj.weight']
    miunsi_state['block.9.attn.q_proj.weight'] = tmp['transformer.h.9.attn.attention.q_proj.weight']
    miunsi_state['block.9.attn.out_proj.weight'] = tmp['transformer.h.9.attn.attention.out_proj.weight']
    miunsi_state['block.9.head.linear1.weight'] = tmp['transformer.h.9.mlp.c_fc.weight']
    miunsi_state['block.9.head.linear2.weight'] = tmp['transformer.h.9.mlp.c_proj.weight']

    # block10
    miunsi_state['block.10.ln.weight'] = tmp['transformer.h.10.ln_1.weight']
    miunsi_state['block.10.attn.ln.weight'] = tmp['transformer.h.10.ln_1.weight']
    miunsi_state['block.10.attn.k_proj.weight'] = tmp['transformer.h.10.attn.attention.k_proj.weight']
    miunsi_state['block.10.attn.v_proj.weight'] = tmp['transformer.h.10.attn.attention.v_proj.weight']
    miunsi_state['block.10.attn.q_proj.weight'] = tmp['transformer.h.10.attn.attention.q_proj.weight']
    miunsi_state['block.10.attn.out_proj.weight'] = tmp['transformer.h.10.attn.attention.out_proj.weight']
    miunsi_state['block.10.head.linear1.weight'] = tmp['transformer.h.10.mlp.c_fc.weight']
    miunsi_state['block.10.head.linear2.weight'] = tmp['transformer.h.10.mlp.c_proj.weight']

    # block11
    miunsi_state['block.11.ln.weight'] = tmp['transformer.h.11.ln_1.weight']
    miunsi_state['block.11.attn.ln.weight'] = tmp['transformer.h.11.ln_1.weight']
    miunsi_state['block.11.attn.k_proj.weight'] = tmp['transformer.h.11.attn.attention.k_proj.weight']
    miunsi_state['block.11.attn.v_proj.weight'] = tmp['transformer.h.11.attn.attention.v_proj.weight']
    miunsi_state['block.11.attn.q_proj.weight'] = tmp['transformer.h.11.attn.attention.q_proj.weight']
    miunsi_state['block.11.attn.out_proj.weight'] = tmp['transformer.h.11.attn.attention.out_proj.weight']
    miunsi_state['block.11.head.linear1.weight'] = tmp['transformer.h.11.mlp.c_fc.weight']
    miunsi_state['block.11.head.linear2.weight'] = tmp['transformer.h.11.mlp.c_proj.weight']

    miunsi_state['final_ln.weight'] = tmp['transformer.ln_f.weight']
    miunsi_state['final_linear.weight'] = tmp['lm_head.weight']



    miunsi.load_state_dict(miunsi_state)

    return miunsi

if __name__ == '__main__':
    gin_bbai()