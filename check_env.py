import sys
print("Python:", sys.version)

packages = {}
checks = [
    ("torch", "import torch; v=torch.__version__; cuda=torch.cuda.is_available(); gpu=torch.cuda.get_device_name(0) if cuda else 'None'"),
    ("numpy", "import numpy as np; v=np.__version__"),
    ("pandas", "import pandas as pd; v=pd.__version__"),
    ("PIL", "from PIL import Image; import PIL; v=PIL.__version__"),
    ("sklearn", "import sklearn; v=sklearn.__version__"),
    ("pyarrow", "import pyarrow; v=pyarrow.__version__"),
    ("cv2", "import cv2; v=cv2.__version__"),
    ("matplotlib", "import matplotlib; v=matplotlib.__version__"),
]

for name, code in checks:
    try:
        exec(code)
        local_v = locals().get('v', '?')
        print(f"[OK] {name}: {local_v}")
        if name == "torch":
            local_cuda = locals().get('cuda', False)
            local_gpu = locals().get('gpu', 'None')
            print(f"     CUDA available: {local_cuda}")
            print(f"     GPU: {local_gpu}")
    except Exception as e:
        print(f"[MISSING] {name}: {e}")
