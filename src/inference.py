import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
onnx_model_path = r"./.models/yolov8s.onnx"  # Change this to your model path
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])


# Preprocess input image
def preprocess(image_path, img_size=640):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    image = np.transpose(image, (2, 0, 1))  # Change HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Run inference
def run_inference(image_path):
    input_tensor = preprocess(image_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    
    return outputs

# Postprocess output (basic implementation)
def postprocess(output, conf_threshold=0.1):
    detections = output[0]  # Adjust based on your model's output structure
    results = []

    for det in detections:
        conf = det[4]  # Extract confidence score

        if isinstance(conf, np.ndarray) and conf.size == 1:
            conf = conf.item()  # Convert single-element array to scalar
        elif isinstance(conf, np.ndarray):  
            conf = conf.max()  # Take the max confidence if multiple

        if conf > conf_threshold:
            x, y, w, h = det[:4]  # Bounding box
            results.append((x, y, w, h, conf))

    return results

# Run script
if __name__ == "__main__":
    image_path = r"./CarEngineBayDataset/images/val/23.jpg"  # Change this to your image path
    output = run_inference(image_path)
    results = postprocess(output)

    print("Detections:", results)