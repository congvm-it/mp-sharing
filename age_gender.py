import tensorflow as tf
import numpy as np

class AgeGender():
    def __init__(self, model_path=None):
        self.model_path = model_path or 'age_gender.tflite'
        self.model = tf.lite.Interpreter(self.model_path)
        self.model.allocate_tensors()
        self.inp_idx = self.model.get_input_details()[0]['index']
        self.age_idx = self.model.get_output_details()[0]['index']
        self.gender_idx = self.model.get_output_details()[1]['index']

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__

    def inference(self, inp):
        inp = inp.astype(np.float32)
        self.model.set_tensor(self.inp_idx, tf.convert_to_tensor(inp))
        self.model.invoke()
        pred_age = self.model.get_tensor(self.age_idx)[0]
        pred_gender = self.model.get_tensor(self.gender_idx)[0]
        return {'pred_age': pred_age, 'pred_gender': pred_gender}
