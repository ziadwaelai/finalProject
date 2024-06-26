import streamlit as st
import warnings
import time
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# Initialize models
blip_model_id = "Salesforce/blip-image-captioning-base"
caption_processor = BlipProcessor.from_pretrained(blip_model_id)
caption_model = BlipForConditionalGeneration.from_pretrained(blip_model_id)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate caption for an image
def generate_caption(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = caption_processor(image, return_tensors="pt").to(caption_model.device)
    output = caption_model.generate(**inputs, max_length=20, num_beams=2)
    caption = caption_processor.decode(output[0], skip_special_tokens=True)
    return caption

# Function to get embeddings for a given text
def get_embedding(text):
    return embedding_model.encode(text, convert_to_tensor=True)

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return util.pytorch_cos_sim(embedding1, embedding2).item()

# Streamlit app
st.title('Semantic Image Search')

# Input for image URL
image_url = st.text_input('Enter the image URL:', '')
query = st.text_input('Enter your search query:', '')

if image_url and query:
    # Display the image
    st.image(image_url, caption='Input Image', use_column_width=True)
    
    # Generate caption
    caption = generate_caption(image_url)
    st.write('Generated Caption:', caption)
    
    # Get embeddings
    caption_embedding = get_embedding(caption)
    query_embedding = get_embedding(query)
    
    # Calculate similarity
    similarity = cosine_similarity(caption_embedding, query_embedding)
    st.write(f'Cosine Similarity: {similarity:.2f}')
    
    # Check if the image belongs to the query
    threshold = 0.5  # You can adjust this threshold
    if similarity > threshold:
        st.write("The image belongs to the query.")
    else:
        st.write("The image does not belong to the query.")
