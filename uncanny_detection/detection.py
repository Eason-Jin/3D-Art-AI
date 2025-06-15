from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
from ultralytics import YOLO
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt


def is_uncanny_1(image_path):
    image = Image.open(image_path).convert("RGB")

    model_id = "HuggingFaceM4/idefics2-8b"

    processor = Idefics2Processor.from_pretrained(model_id)
    model = Idefics2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    prompt = "You are given an image. Evaluate whether it is canny or uncanny. An image is uncanny if it evokes a sense of visual unease or unnaturalness—this may include distorted anatomy, unresolved form-function tension, eerie symmetries, or features that imply hybrid or parasitic use. Consider if the object suggests former identities (like residual limbs or ancestral shapes), broken networks, or docking points hinting at unknown systems. Ask whether the image might transform under motion or shifting light, and whether human response—trust, fear, or indifference—might be misleading. If the image appears visually coherent, natural, and free from such ambiguity, it is canny. Respond with only one word: 'CANNY' or 'UNCANNY'."

    inputs = processor(text=prompt, images=image,
                       return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10)
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0].strip().lower()

    print("Model response:", response)


def is_uncanny_2(image_path):
    model = YOLO('yolov8s.pt')
    results = model(image_path)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        print(f"{class_name}: {confidence:.2f}")

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
            print("Image is CANNY")

    rendered_image = results.plot()  # Returns an image array (BGR format)
    image_rgb = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()


image_path = "images/image13.jpeg"
# is_uncanny_1(image_path)
is_uncanny_2(image_path)
