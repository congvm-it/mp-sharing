from shared_manager import SharedManager
from age_gender import AgeGender
import numpy as np
from queue import Queue
from multiprocessing.queues import SimpleQueue
import cv2

# shared_manager = SharedManager()
# shared_manager.register(AgeGender, 'AgeGender1', model_path='age_gender.tflite')
# shared_manager.register(AgeGender, 'AgeGender2', model_path='age_gender.tflite')
# shared_manager.register(AgeGender, 'AgeGender3', model_path='age_gender.tflite')
# shared_manager.allocate_memory()


# inp = np.ones((1, 128, 128, 3))
# ag1 = shared_manager['AgeGender1']
# ag2 = shared_manager['AgeGender2']
# ag3 = shared_manager['AgeGender3']

# print(ag1.inference(inp))
# print(ag2.inference(inp))
# print(ag3.inference(inp))


img_arr = cv2.imread('/home/congvm/Downloads/e24bb0d44ad8b786eec9.jpg')[..., ::-1]
queue = SimpleQueue()

queue.put(img_arr)
queue.put(img_arr)