from vllm import LLM, SamplingParams
# Define prompts
CUDA_LAUNCH_BLOCKING = 1
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.1, max_tokens=50)

# Initialize the vLLM engine
# llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", dtype='half')

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    dtype="half",
    gpu_memory_utilization = 0.95,
    max_model_len=1000,
    enforce_eager=True
)
# Generate outputs
outputs = llm.generate(prompts, sampling_params)

# Print generated results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}\n")
