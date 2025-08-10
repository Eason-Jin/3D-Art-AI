from transformers import AutoProcessor, AutoModelForVision2Seq
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from reinforcement_learning.utils import load_images, UNCANNY_FOLDER, NOT_UNCANNY_FOLDER, calculate_confusion_matrix
import os
from render_model import render_model

os.environ["HF_HOME"] = "/data/ejin458/huggingface"

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def is_uncanny_vlm(image):
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceM4/idefics2-8b",).to(DEVICE)

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

    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)
    response = response[0]
    filtered_response = response[response.find("Assistant: ")+10:-1].upper()
    print("Model response:", filtered_response)
    return filtered_response == "UNCANNY"


def is_uncanny_yolo(image, display=False):
    model = YOLO('yolo11x.pt')
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
        # Convert BGR to RGB for display
        image_rgb = rendered_image[:, :, ::-1]

        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()


def main():
    uncanny_images = load_images(UNCANNY_FOLDER, True)
    not_uncanny_images = load_images(NOT_UNCANNY_FOLDER, False)

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for element in uncanny_images + not_uncanny_images:
        image = element['image']
        is_uncanny = element['is_uncanny']
        result = is_uncanny_vlm(image)
        print(f"Actual: {is_uncanny}")
        if result and is_uncanny:
            # Model response is "UNCANNY" and is_uncanny is True
            true_positive += 1
            print("Incrementing true positive")
        elif result and not is_uncanny:
            # Model response is "UNCANNY" but is_uncanny is False
            false_positive += 1
            print("Incrementing false positive")
        elif not result and is_uncanny:
            # Model response is "NOT UNCANNY" but is_uncanny is True
            false_negative += 1
            print("Incrementing false negative")
        else:
            # Model response is "NOT UNCANNY" and is_uncanny is False
            true_negative += 1
            print("Incrementing true negative")

    accuracy, precision, recall = calculate_confusion_matrix(true_positive, false_positive, true_negative, false_negative)

    print("\nConfusion Matrix:")
    print(
        f"TP: {true_positive}\tFP: {false_positive}\nFN: {false_negative}\tTN: {true_negative}")
    print(
        f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

def judge_model(model_path):
    renders = render_model(model_path)
    uncanny_count = 0
    for render in renders:
        is_uncanny = is_uncanny_vlm(render)
        if is_uncanny:
            uncanny_count += 1
    if uncanny_count > len(renders) / 2:
        print(f"The model {model_path} is considered to be UNCANNY.")
    else:
        print(f"The model {model_path} is considered to be NOT UNCANNY.")

if __name__ == "__main__":
    # main()
    judge_model("obj/0.glb")
