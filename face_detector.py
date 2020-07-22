import onnxruntime as ort
from vision.utils import box_utils_numpy as box_utils
import numpy as np
import cv2
import time


class LightweightFaceDetector():
    def __init__(self, model_path, gpu_id=-1, mem_fraction=0.2, input_size=(320, 240), intra_op_num_threads=1):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.image_mean = np.array([127, 127, 127])
        self.std = 128.0
        self.intra_op_num_threads = intra_op_num_threads
        self.input_size = input_size
        self._load_model(model_path, gpu_id)

    def _load_model(self, onnx_path, gpu_id):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.intra_op_num_threads > 1:
            exec_mode = ort.ExecutionMode.ORT_PARALLEL
        else:
            exec_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.execution_mode = exec_mode
        sess_options.intra_op_num_threads = self.intra_op_num_threads
        self.ort_session = ort.InferenceSession(onnx_path, sess_options)
        self.input_name = self.ort_session.get_inputs()[0].name

    def _preprocess(self, img_arr):
        img_arr = cv2.resize(img_arr, self.input_size)
        img_arr = (img_arr - self.image_mean) / self.std
        img_arr = np.transpose(img_arr, [2, 0, 1])
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = img_arr.astype(np.float32)
        return img_arr

    def _predict(self, img_arr):
        confidences, boxes = self.ort_session.run(
            None, {self.input_name: img_arr})
        return confidences, boxes

    def detect(self, img_arr, threshold=0.8):
        """Main interface"""
        preprocessed_img_arr = self._preprocess(img_arr)
        confidences, boxes = self._predict(preprocessed_img_arr)
        bboxes, labels, probs = self._postprocess(img_arr.shape[1],
                                                  img_arr.shape[0],
                                                  confidences,
                                                  boxes,
                                                  threshold)
        return bboxes, labels, probs

    def _postprocess(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate(
                [subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                           iou_threshold=iou_threshold,
                                           top_k=top_k,
                                           )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])

        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]
