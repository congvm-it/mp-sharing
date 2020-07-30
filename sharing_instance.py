# shared_model_between_processes.py
# Shared DL Model
# congvm
import time
from shared_manager import SharedManager
from face_detector import LightweightFaceDetector
from age_gender import AgeGender
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from multiprocessing import Queue
from multiprocessing import Lock


shared_manager = SharedManager()
shared_manager.register(LightweightFaceDetector,
                        'version-RFB-320-optimized-graph.onnx')
shared_manager.register(AgeGender, 'age_gender.tflite')

all_shared_instances = shared_manager.get_shared_instances()

inp = np.ones((1, 128, 128, 3))

inp3 = np.ones((128, 128, 3))


age_gender = all_shared_instances['AgeGender']
face_detector = all_shared_instances['LightweightFaceDetector']
# print(age_gender.inference(inp))


def callback_fn(idx, result):
    print(idx, result)


def predict(model, inp, l, idx, callback):
    print('start:', idx)
    with l:
        result = model.detect(inp)
    print(result)
    callback(idx, result)


executor = ProcessPoolExecutor(max_workers=4)
for idx in range(4):
    future = executor.submit(predict, face_detector, inp, l, idx, callback_fn)


while True:
    time.sleep(1)


# ONNX: Not process-safe, require lock
# TFLite: Not process-safe, require lock
