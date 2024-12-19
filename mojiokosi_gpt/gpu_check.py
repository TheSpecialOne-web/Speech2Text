import torch

'''
 pip3 install torch torchvision torchaudio --index-url
  https://download.pytorch.org/whl/cu118
'''

# GPUが利用可能かどうかをチェック
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available.")

print(f"PyTorch CUDA バージョン: {torch.version.cuda}")
print(f"PyTorch ビルド CUDA バージョン: {torch.backends.cudnn.version()}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"GPU 名: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
