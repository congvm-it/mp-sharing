# shared_model_between_processes.py
# Shared DL Model
# congvm

from shared_manager import SharedManager
from face_detector import LightweightFaceDetector
from age_gender import AgeGender
import numpy as np


shared_manager = SharedManager()
shared_manager.register(LightweightFaceDetector,
                        'version-RFB-320-optimized-graph.onnx')
shared_manager.register(AgeGender, 'age_gender.tflite')

all_shared_instances = shared_manager.get_shared_instances()

inp = np.ones((1, 128, 128, 3))
age_gender = all_shared_instances['AgeGender']
print(age_gender.inference(inp))
