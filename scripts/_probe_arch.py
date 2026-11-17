import json
d = json.load(open(r'D:\_SATIN_AI_2\beellama\beehive-cache\manifest.json'))
# find one attn_k and attn_v dims
for sub in ['attn_k.weight', 'attn_v.weight', 'attn_output.weight']:
    for l in d['layers']:
        if l['name'].endswith(sub) and l['name'].startswith('blk.'):
            print(l['name'], l['dims'])
            break
# n_layer: max distinct block.TOKEN_EMBD count
# KV bytes = 2 * n_layer * n_kv_heads * head_dim * ctx * (2 bytes fp16)
# infer from attn_output.weight (n_embd out) and attn_k (n_kv*head_dim, n_embd)
# If attn_k.weight is [n_kv*head_dim, 2048] and attn_q [2048,2048] => ratio gives n_kv