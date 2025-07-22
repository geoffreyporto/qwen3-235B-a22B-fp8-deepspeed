#!/bin/bash
# setup_rocm_env.sh - Instala PyTorch ROCm + Transformers + DeepSpeed
echo "🔧 Instalando PyTorch con ROCm..." 
pip install --upgrade pip setuptools wheel
# Instalar PyTorch y librerías con soporte ROCm (ajustar si ya estás en ROCm 6.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
echo "✅ PyTorch ROCm instalado."
echo "🔧 Instalando Hugging Face Transformers, Accelerate y DeepSpeed..."
pip install transformers accelerate deepspeed
echo "✅ Entorno ROCm configurado correctamente."
