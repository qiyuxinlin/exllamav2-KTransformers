from exllamav2 import ExLlamaV2, ExLlamaV2Config,ExLlamaV2DeppSeekCache
from exllamav2.linear import ExLlamaV2Linear
import torch
# Load a model
import os 
os.environ['CUDA_VISIBLE_DEVICES']= '2,3'
model_dir = "/data/model/DeepSeek-V2-Lite-gptq-4bit"
config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
cache = ExLlamaV2DeppSeekCache(model, max_seq_len = 4096, lazy = True)
model.load_autosplit(cache, progress = True)

# Grab a linear module

module = model.modules_dict["model.layers.0.self_attn.kv_b_proj"]

# Dimensions and split point
hidden_dim = 128
num_heads = 16
kv_lora_rank = 512
m = module.in_features
n = module.out_features
n_a = hidden_dim
n_b = n - num_heads*hidden_dim

# Get quantized tensors for split matrices

orig_weights = module.load_weight()

qweight = orig_weights["qweight"].reshape(orig_weights["qweight"].shape[0],num_heads,-1)
qweight_a = qweight[:,:, :n_a].reshape(qweight.shape[0],-1)
qweight_b = qweight[:,:, n_a:].reshape(qweight.shape[0],-1)

qzeros = orig_weights["qzeros"].reshape(orig_weights["qzeros"].shape[0],num_heads,-1)
qzeros_a = qzeros[:,:, :n_a // 8].reshape(qzeros.shape[0],-1)
qzeros_b = qzeros[:,:, n_a // 8:].reshape(qzeros.shape[0],-1)

scales = orig_weights["scales"].reshape(orig_weights["scales"].shape[0],num_heads,-1)
scales_a = scales[:,:, :n_a].reshape(scales.shape[0],-1)
scales_b = scales[:,:, n_a:].reshape(scales.shape[0],-1)

g_idx = orig_weights["g_idx"]
g_idx_a = g_idx
g_idx_b = g_idx

# Create new linear modules with split tensors

module_a = ExLlamaV2Linear(model, "dummy_key_a", m, num_heads*hidden_dim, False)
module_a.set_device_idx(0)
module_a.load({
    "qweight": qweight_a,
    "qzeros": qzeros_a,
    "scales": scales_a,
    "g_idx": g_idx_a,
})


# Compare against original
module.load()
original = module.get_weight_tensor_dq().T.view(num_heads, -1, kv_lora_rank)
original = original[:, :hidden_dim, :].reshape(-1, kv_lora_rank).T
split_a = module_a.get_weight_tensor_dq()
print(torch.allclose(original, split_a))