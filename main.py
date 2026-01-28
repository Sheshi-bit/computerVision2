# Install Dependency (run once)
# !pip install groq transformers torch streamlit pillow

from groq import Groq
import streamlit as st
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch

# -------------------------------
# Initialize Groq Client
# -------------------------------
llm = Groq(api_key="YOUR_GROQ_API_KEY")  # DO NOT hardcode in real projects

# -------------------------------
# Dynamically pick a valid Groq model
# -------------------------------
def pick_groq_model(client):
    models = client.models.list().data

    # Priority order: chat-capable, larger models first
    priority_keywords = ["chat", "llama", "8b", "instant"]

    for keyword in priority_keywords:
        for m in models:
            if keyword.lower() in m.id.lower():
                return m.id

    # Absolute fallback
    return models[0].id


GROQ_MODEL = pick_groq_model(llm)
print(f"Using Groq model: {GROQ_MODEL}")

# -------------------------------
# Load YOLOs model
# -------------------------------
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("YOLOs Object Detection with Groq Summarization")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Detecting objects...")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs,
        threshold=0.9,
        target_sizes=target_sizes
    )[0]

    detection_results = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        detection_results.append(
            f"{model.config.id2label[label.item()]} "
            f"(confidence={round(score.item(), 3)}, box={box.tolist()})"
        )

    # -------------------------------
    # Send results to Groq
    # -------------------------------
    groq_prompt = (
        "Summarize the following object detection results clearly and concisely:\n"
        + "\n".join(detection_results)
    )

    response = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You summarize computer vision object detection results."
            },
            {
                "role": "user",
                "content": groq_prompt
            }
        ]
    )

    summary = response.choices[0].message.content

    st.subheader("Summary of Detection Results")
    st.write(summary)

    # -------------------------------
    # Draw bounding boxes
    # -------------------------------
    draw = ImageDraw.Draw(image)

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        draw.text(
            (box[0], box[1]),
            f"{model.config.id2label[label.item()]} ({round(score.item(), 2)})",
            fill="red"
        )

    st.image(image, caption="Detected Objects", use_column_width=True)
