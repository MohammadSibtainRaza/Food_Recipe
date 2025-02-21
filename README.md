# Food Image to Recipe Generator

## Overview
The **Food Image to Recipe Generator** is an AI-powered web application that classifies food images and generates detailed recipes based on the classification. The project utilizes **OpenAI's CLIP model** for image classification and **Google Gemini API** for recipe generation. The UI is built with **Streamlit** and the app is deployed on **Streamlit Community Cloud**.

## Features
- **Upload a food image** (.jpg/.jpeg/.png)
- **Classify the food item** using CLIP (ViT-B/32)
- **Generate a detailed recipe** including ingredients, preparation time, and cooking instructions
- **Interactive Streamlit UI** for seamless user experience

## Workflow Steps
### **Step 1: Image Classification with CLIP**
1. User uploads a food image via the Streamlit UI.
2. Initially attempted to use a Kaggle dataset for classification but faced credential issues.
3. Switched to manual labels: `pizza, burger, pasta, sushi, salad, cake`.
4. CLIP model (**ViT-B/32**) classifies the image based on these labels.

### **Step 2: Loading Models**
1. Used Hugging Face’s **transformers** library to load the CLIP model and processor.
2. Utilized `@st.cache_resource` to cache the model and improve app performance.

### **Step 3: Recipe Generation using Google Gemini API**
1. Initially used OpenAI API but faced request exhaustion and payment limits.
2. Switched to Hugging Face, but encountered access issues with private repositories.
3. Final implementation used **Google Gemini API (Bard)** for recipe generation.
4. AI generates **detailed recipes** including ingredients, prep time, and instructions.

### **Step 4: Streamlit UI Implementation**
1. Developed an **interactive UI** where users can:
   - Upload food images (.jpg/.jpeg/.png).
   - View the **classified food label**.
   - See the **AI-generated recipe**.
2. Example: **Uploaded Burger.jpg → Generated Burger Recipe**.

### **Step 5: Deployment**
- Deployed the app using **Streamlit Community Cloud** for public access via **GitHub**.

## Challenges Faced & Solutions
| Challenge | Solution |
|-----------|----------|
| Kaggle Dataset Access Issues | Switched to manual labels |
| API Limitations (OpenAI) | Tried Hugging Face, finalized with Google Gemini API |
| Model Access Issues | Moved to Gemini API after Hugging Face restrictions |

## Demo & Code
- **Demo Link:** [Food Image to Recipe Generator](https://foodrecipe-dsuft7fkohwmbai76wfkzj.streamlit.app/)
- **GitHub Repository:** [Food Recipe Generator](https://github.com/MohammadSibtainRaza/Food_Recipe/tree/main)

## Installation & Setup
### **1. Clone the Repository**
```bash
git clone https://github.com/MohammadSibtainRaza/Food_Recipe.git
cd Food_Recipe
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit App**
```bash
streamlit run app.py
```

## Resources
1. [Streamlit Documentation](https://docs.streamlit.io/)
2. [Streamlit Community Cloud (Deployment)](https://streamlit.io/cloud)
3. [CLIP Model on Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)
4. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
5. [Google AI Studio (Gemini API)](https://aistudio.google.com/app)
6. [OpenAI API Documentation](https://platform.openai.com/docs)
7. [Hugging Face Hub](https://huggingface.co/)
8. [Kaggle Datasets](https://www.kaggle.com/datasets)
9. [Pillow Documentation](https://pillow.readthedocs.io/en/stable/)
10. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
11. [Transformers Docs](https://huggingface.co/docs/transformers/index)

## License
This project is licensed under the **MIT License**.

