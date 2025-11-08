import gradio as gr
from transformers import pipeline
import os
import warnings

# Suppress warnings and logging
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Reduce transformers logging
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Option 1: Medical Image Classification
# Using a general vision model (you can replace with medical-specific model)
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Option 2: NCBI Disease Text Classification
# Load NCBI disease model for text classification
try:
    text_classifier = pipeline("text-classification", 
                              model="ugaray96/biobert_ncbi_disease_ner",
                              tokenizer="ugaray96/biobert_ncbi_disease_ner")
except:
    # Fallback to a general biomedical text model
    text_classifier = pipeline("text-classification",
                              model="dmis-lab/biobert-base-cased-v1.1")

def classify_image(image):
    """Classify medical images"""
    if image is None:
        return "Please upload an image"
    
    try:
        results = image_classifier(image)
        
        # Format top 3 results
        output = "Image Classification Results:\n"
        for i, result in enumerate(results[:3], 1):
            label = result['label']
            confidence = result['score'] * 100
            output += f"{i}. {label} - {confidence:.1f}%\n"
        
        return output
    except Exception as e:
        return f"Error processing the image: {str(e)}. Please ensure the image is in a supported format (JPEG, PNG, BMP)."

def classify_text(text):
    """Classify medical text for disease mentions"""
    if not text or text.strip() == "":
        return "Please enter some medical text"
    
    try:
        results = text_classifier(text)
        
        if isinstance(results, list) and len(results) > 0:
            output = "Disease Classification Results:\n"
            for i, result in enumerate(results[:3], 1):
                label = result['label']
                confidence = result['score'] * 100
                output += f"{i}. {label} - {confidence:.1f}%\n"
        else:
            output = "No disease classifications found"
        
        return output
    except Exception as e:
        return f"Error in text classification: {str(e)}"

# Create Gradio interface with tabs
with gr.Blocks(title="Medical Classifier") as app:
    gr.Markdown("# Medical Classification Tool")
    gr.Markdown("Choose between image classification or text analysis for disease mentions")
    
    with gr.Tabs():
        # Tab 1: Image Classification
        with gr.TabItem("Image Classification"):
            gr.Markdown("### Upload a medical image for classification")
            
            with gr.Row():
                image_input = gr.Image(type="pil", label="Upload Medical Image")
                image_output = gr.Textbox(label="Classification Results", lines=6)
            
            image_btn = gr.Button("Classify Image", variant="primary")
            image_btn.click(
                fn=classify_image,
                inputs=image_input,
                outputs=image_output
            )
        
        # Tab 2: Text Classification (NCBI Disease)
        with gr.TabItem("Text Analysis (NCBI Disease)"):
            gr.Markdown("### Enter medical text for disease entity recognition")
            
            with gr.Row():
                text_input = gr.Textbox(
                    label="Medical Text", 
                    placeholder="Enter medical text, symptoms, or clinical notes here...",
                    lines=4
                )
                text_output = gr.Textbox(label="Disease Analysis Results", lines=6)
            
            text_btn = gr.Button("Analyze Text", variant="primary")
            text_btn.click(
                fn=classify_text,
                inputs=text_input,
                outputs=text_output
            )
    
    # Instructions
    gr.Markdown("""
    ## Instructions:
    
    **Image Classification Tab:**
    - Upload any medical image (X-ray, CT scan, etc.)
    - The model will attempt to classify the image content
    - Note: For specialized medical diagnosis, use domain-specific models
    
    **Text Analysis Tab:**
    - Enter medical text, symptoms, or clinical descriptions
    - The NCBI disease model will identify disease entities
    - This is useful for processing clinical notes or medical literature
    
    ## Installation Requirements:
    ```bash
    pip install gradio transformers torch pillow
    ```
    """)

if __name__ == "__main__":
    # Launch with optimized settings to reduce warnings
    app.launch(
        server_name="127.0.0.1",  # More specific than localhost
        server_port=7860,
        share=False,
        inbrowser=True,  # Auto-open browser
        quiet=True,      # Reduce console output
        show_error=True, # Still show important errors
        favicon_path=None,  # Prevent favicon 404s
        ssl_verify=False    # For local development
    )
    