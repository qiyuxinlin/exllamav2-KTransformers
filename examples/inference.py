
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer, ExLlamaV2DeppSeekCache
from exllamav2.generator import ExLlamaV2DynamicGenerator
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# model_dir = "/data/model/Mistral-87B-Instruct-v0.1"
# model_dir = "/data/model/DeepSeek-V2-Lite-Chat"
# model_dir = "/data/model/DeepSeek-V2-Lite-gptq-4bit"
# model_dir = "/data/model/Qwen1.5-MoE-A2.7B-Chat"
# model_dir = "/data/model/Qwen1.5-7B-Chat"
# model_dir = "/data/model/Qwen2-57B-A14B-Instruct"
model_dir = "/data/model/Qwen2-57B-A14B-Instruct-GPTQ-Int4" 
config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len = 4096, lazy = True)
# cache = ExLlamaV2DeppSeekCache(model, max_seq_len = 4096, lazy = True)
model.load_autosplit(cache, progress = True)

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize the generator with all default parameters

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
    paged = False,
)

max_new_tokens = 250

# Warmup generator. The function runs a small completion job to allow all the kernels to fully initialize and
# autotune before we do any timing measurements. It can be a little slow for larger models and is not needed
# to produce correct output.

# generator.warmup()

# Generate one completion, using default settings
# prompt = "<im_start>user\n你好，请问你能做什么?<im_end>\n<im_start>assistant\n"
prompt = "你好，请问你能做什么"
# prompt = "<｜begin▁of▁sentence｜>User: 你好，请问你能做什么\n\nAssistant:"

with Timer() as t_single:
    output = generator.generate(prompt = prompt, max_new_tokens = max_new_tokens, add_bos = False, encode_special_tokens=True,decode_special_tokens=True,stop_conditions=['<im_end>',151645])

print("-----------------------------------------------------------------------------------")
print("- Single completion")
print("-----------------------------------------------------------------------------------")
print(output)
print()

# Do a batched generation

# prompts = ['你好，请问你能做什么']
prompts = [
    "Once upon a time,",
    "The secret to success is",
    "There's no such thing as",
    "Here's why you should adopt a cat:",
]

with Timer() as t_batched:
    outputs = generator.generate(prompt = prompts, max_new_tokens = max_new_tokens, add_bos = True)

for idx, output in enumerate(outputs):
    print("-----------------------------------------------------------------------------------")
    print(f"- Batched completion #{idx + 1}")
    print("-----------------------------------------------------------------------------------")
    print(output)
    print()

print("-----------------------------------------------------------------------------------")
print(f"speed, bsz 1: {max_new_tokens / t_single.interval:.2f} tokens/second")
print(f"speed, bsz {len(prompts)}: {max_new_tokens * len(prompts) / t_batched.interval:.2f} tokens/second")
