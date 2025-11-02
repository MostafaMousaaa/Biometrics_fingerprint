import numpy as np
import cv2

class KerasClassifier:
    def __init__(self, model_path: str):
        try:
            import tensorflow as tf
        except Exception as e:
            raise RuntimeError("TensorFlow is required for KerasClassifier") from e
        self.tf = tf
        self.model = tf.keras.models.load_model(model_path)

    def _prep(self, img: np.ndarray, size=(224,224), grayscale=True, norm01=True) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if grayscale else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2 and not grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if grayscale:
            x = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            x = x.astype(np.float32)
            if norm01:
                x /= 255.0
            x = x[..., None]  # HWC1
        else:
            x = cv2.resize(img, size, interpolation=cv2.INTER_AREA).astype(np.float32)
            if norm01:
                x /= 255.0
        x = np.expand_dims(x, 0)  # NHWC
        return x

    def predict_image(self, img: np.ndarray, size=(224,224), grayscale=True, norm01=True) -> float:
        x = self._prep(img, size=size, grayscale=grayscale, norm01=norm01)
        y = self.model.predict(x, verbose=0)
        y = np.array(y)
        # handle [N,1] sigmoid or [N,2] softmax
        if y.ndim == 2 and y.shape[1] == 1:
            p = float(y[0,0])
        elif y.ndim == 2 and y.shape[1] >= 2:
            p = float(y[0,1])  # assume index 1 is "Real"
        else:
            p = float(y.ravel()[0])
        p = float(np.clip(p, 0.0, 1.0))
        return p
