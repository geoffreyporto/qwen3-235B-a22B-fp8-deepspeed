from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# Ruta local o repo Hugging Face
model_name = "/workspace/models/Qwen3-235B-A22B" 
# O usa: "Qwen/Qwen3-235B-A22B" 
print(f"🚀 Cargando modelo desde {model_name}")

# Cargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Cargar el modelo con DeepSpeed + ROCm
model = AutoModelForCausalLM.from_pretrained( model_name, 
	torch_dtype="auto", # ROCm: usa FP16 automáticamente
	low_cpu_mem_usage=True,
	trust_remote_code=True,
	device_map="auto" # Usa la MI300X automáticamente
)

print("✅ Modelo cargado en GPU AMD.")

# Crear pipeline de generación de texto
generator = pipeline( "text-generation",
	model=model,
	tokenizer=tokenizer,
	device=0 # GPU AMD visible
)

# Probar el modelo con un prompt de ejemplo
prompt = "Resumen ejecutivo sobre la gestión financiera moderna."
outputs = generator(prompt, max_new_tokens=512)
print("📄 Resultado:"
print(outputs[0]['generated_text'])
