import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai

# Configure Google Gemini API
genai.configure(api_key="AIzaSyCgYwE3W9q07k8QbIKGI5kmHGkVZiiiAhM")

# Load CLIP model and processor
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Classify food image
def classify_food(image, model, processor):
    labels = ["pizza", "burger", "pasta", "sushi", "salad", "cake"]
    inputs = processor(images=image, text=labels, return_tensors="pt")
    outputs = model(**inputs)
    predicted_idx = torch.argmax(outputs.logits_per_image).item()
    return labels[predicted_idx]

# Generate recipe using Gemini API
def generate_recipe_gemini(food_item):
    if food_item == "Unknown":
        return "Could not classify the food item. Please try another image."
    
    prompt = f"Generate a detailed recipe for {food_item}. Include ingredients, preparation time, cooking instructions, and serving suggestions."
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")  # Using a free model
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating recipe: {str(e)}"

# Streamlit UI
st.title("Food Image to Recipe Generator")
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    model, processor = load_model()
    food_item = classify_food(image, model, processor)
    st.write(f"Classified Food Item: **{food_item}**")
    recipe = generate_recipe_gemini(food_item)
    st.write("Generated Recipe:")
    st.write(recipe)
