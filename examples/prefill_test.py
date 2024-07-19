
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
from util import format_prompt, get_stop_conditions
from blessed import Terminal

model_dir = "/data/model/Qwen1.5-MoE-A2.7B-Chat"
config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len = 32768, lazy = True)
model.load_autosplit(cache, progress = True)

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize the generator with all default parameters

max_chunk_size = 2048
max_new_tokens = 500

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
    max_chunk_size = max_chunk_size,
    max_batch_size = 20,
    paged = True
)

# Warmup generator. The function runs a small completion job to allow all the kernels to fully initialize and
# autotune before we do any timing measurements. It can be a little slow for larger models and is not needed
# to produce correct output.

generator.warmup()

# Generate one completion, using default settings

prompts_list = [
    "Once upon a time, there is a place.", 
    "Can you guess the next number in this sequence: " + ", ".join(str(n) for n in range(500)),
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 123 else 69) for n in range(1000)),
    "Please guess the next 20 numbers in this sequence: " + ", ".join(str(n) for n in range(700)),
]

# Generate and Enqueue jobs
for prompt in prompts_list:

    if isinstance(prompt, list):
        prompts = prompt
        filters = None
    else:
        prompts = [prompt]
        filters = [None]

    if filters is None:
        filters = [None] * len(prompts)
    else:
        assert len(filters) == len(prompts) and \
            all((f is None or isinstance(f, list)) for f in filters), \
            "If using filters, must provide one filter list (or None-value) per prompt."

    prompts = prompt if isinstance(prompt, list) else [prompt]
    batch_size = len(prompts)
    jobs = []
    for idx, p in enumerate(prompts):

        if isinstance(p, str):
            input_ids = generator.tokenizer.encode(p, encode_special_tokens = False, add_bos = True)
        elif isinstance(p, tuple):
            input_ids = [generator.tokenizer.encode(p_, encode_special_tokens = False, add_bos = True) for p_ in p]
        else:
            assert False, "Unexpected type in prompt"


        p_settings = ExLlamaV2Sampler.Settings()

        seed = None

        job = ExLlamaV2DynamicJob(
            input_ids = input_ids,
            max_new_tokens = max_new_tokens,
            min_new_tokens = 50,
            seed = seed,
            stop_conditions = get_stop_conditions("llama", tokenizer),
            gen_settings = p_settings,
            filters = filters[idx] or [],
            filter_prefer_eos = True,
            token_healing = False,
            decode_special_tokens = False,
        )

        if seed is not None: seed += 1

        # serial = generator.enqueue(job)
        jobs.append(job)

    # Collect outputs until all jobs finish
    generator.enqueue(jobs)

    completions = [""] * batch_size
    last_results = [None] * batch_size

    term = Terminal()

    prefill_token_num = {}

    while generator.num_remaining_jobs():
        results = generator.iterate()
        for r in results:
            if r["stage"] == 'prefill':
                job = r["job"]
                in_prompt = tokenizer.decode(job.sequences[0].input_ids.torch(), decode_special_tokens = True)[0]
                prefill_token_num[job] = r['max_progress']
            # if r["stage"] == "streaming":
            #     print(term.red(r["text"]))
            if r["stage"] == "streaming" and r["eos"]:
                job = r["job"]
                in_prompt = tokenizer.decode(job.sequences[0].input_ids.torch(), decode_special_tokens = True)[0]
                print("\n")
                print(term.black("Input: "))
                print(term.yellow(in_prompt))
                print(term.black("Output:"))
                print(term.red(r["full_completion"]))
                print(term.black("New tokens:    ") + term.green(f"{r['new_tokens']:7} t"))
                print(term.black("Time Prefill:       ") + term.blue(f"{r['time_prefill']:7.2f} s"))
                print(term.black("Time Generate:       ") + term.blue(f"{r['time_generate']:7.2f} s"))
                print(term.black("Prefill tokens:    ") + term.green(f"{prefill_token_num[job]:7} t"))
