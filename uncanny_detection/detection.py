from transformers import AutoProcessor, AutoModelForVision2Seq
from ultralytics import YOLO
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
from reinforcement_learning.utils import load_images, UNCANNY_FOLDER, NOT_UNCANNY_FOLDER

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def is_uncanny_vlm(image):
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
    model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b",).to(DEVICE)
    
    rule = "You are an expert in visual psychology and artistic analysis. Examine the following image and determine whether it feels uncanny or not uncanny. A visual is considered uncanny if it evokes a sense of unease, eeriness, or something being subtly 'off', even if the image is artistic or surreal. If the image is strange or abstract but still feels natural, pleasing, or intentionally stylised, it is not uncanny. Respond with only: 'uncanny' or 'not uncanny'."
    
    messages = [
	    {
	        "role": "user",
	        "content": [
	            {"type": "image"},
	            {"type": "text", "text": rule},
            ]
        },  
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)
    response = response[0]
    filtered_response = response[response.find("Assistant: ")+10:-1].upper()
    # print("Model response:", filtered_response)
    return filtered_response == "UNCANNY"


def is_uncanny_yolo(image_path, display = False):
    model = YOLO('yolo11x.pt')
    image = cv2.imread(image_path)
    results = model(image)[0]
    

    if len(results.boxes) == 0:
        print("Image is UNCANNY (no detections)")
    else:
        confidence_threshold = 0.4
        low_conf_count = sum(
            float(box.conf[0]) < confidence_threshold for box in results.boxes)
        low_conf_ratio = low_conf_count / len(results.boxes)
        if low_conf_ratio > 0.3:
            print("Image is UNCANNY (high low confidence ratio)")
        else:
            print("Image is NOT UNCANNY")

    if display:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            print(f"{class_name}: {confidence:.2f}")
            
        rendered_image = results.plot()  # Returns an image array (BGR format)
        image_rgb = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)
    
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

def main():
    uncanny_images = load_images(UNCANNY_FOLDER, True)
    not_uncanny_images = load_images(NOT_UNCANNY_FOLDER, False)

    correct_count = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    
    for image in uncanny_images:
        is_uncanny = is_uncanny_vlm(image)
        # is_uncanny_yolo(image_path, display = False)
        if is_uncanny:
            correct_count += 1
            true_positive += 1
        else:
            false_negative += 1
    for image in not_uncanny_images:
        is_uncanny = is_uncanny_vlm(image)
        if not is_uncanny:
            correct_count += 1
        else:
            false_positive += 1
    
    accuracy = correct_count / (len(uncanny_images) + len(not_uncanny_images))
    precision = (true_positive / (true_positive + false_positive)) if (true_positive +
    false_positive) > 0 else 0
    recall = (true_positive / (true_positive + false_negative)) if (true_positive + false_negative) > 0 else 0
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

if __name__ == "__main__":
    main()
