import torch
print(torch.cuda.is_available())  # 输出True则正常
print(torch.cuda.device_count())  # 输出1（RTX3050）
print(torch.cuda.get_device_name(0))  # 输出RTX 3050