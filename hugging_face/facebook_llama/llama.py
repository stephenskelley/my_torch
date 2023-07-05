from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
generator = pipeline(model="decapoda-research/llama-7b-hf", device='mps')
print(generator("I can't believe you did such a "))

