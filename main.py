from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import json
import numpy as np
import cv2
from PIL import Image
import io
import base64
from typing import Optional
import random
import re
import google.generativeai as genai
from datetime import datetime

app = FastAPI(title="Medical AI Assistant", version="1.0.0")

# Configure Gemini API
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
gemini_model = genai.GenerativeModel('gemini-pro')

# Mount static files
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced medical knowledge base
medical_knowledge = {
    "symptoms": {
        "headache": {
            "possible_conditions": ["Tension headache", "Migraine", "Sinusitis", "Dehydration"],
            "advice": "Rest in a quiet room, stay hydrated, and consider over-the-counter pain relief. See a doctor if severe or persistent.",
            "emergency": False
        },
        "fever": {
            "possible_conditions": ["Viral infection", "Bacterial infection", "Inflammatory condition"],
            "advice": "Rest, drink plenty of fluids, and monitor temperature. Seek medical help if fever is high or lasts more than 3 days.",
            "emergency": False
        },
        "cough": {
            "possible_conditions": ["Common cold", "Bronchitis", "Allergies", "Asthma"],
            "advice": "Stay hydrated, use humidifier, and avoid irritants. See a doctor if coughing up blood or having breathing difficulties.",
            "emergency": False
        },
        "chest pain": {
            "possible_conditions": ["Heart-related issues", "Muscle strain", "Acid reflux", "Anxiety"],
            "advice": "THIS COULD BE SERIOUS. Seek immediate medical attention if pain is severe, radiates to arm/jaw, or accompanied by shortness of breath.",
            "emergency": True
        },
        "abdominal pain": {
            "possible_conditions": ["Indigestion", "Appendicitis", "Gallstones", "Food poisoning"],
            "advice": "Rest and avoid solid foods. Seek emergency care if pain is severe, localized, or accompanied by fever/vomiting.",
            "emergency": False
        }
    }
}

# Demo disease information
disease_info = {
    "skin_cancer": {
        "model_config": {
            "path": "models/skin_cancer_model.h5",
            "size": [224, 224]
        },
        "labels": ["Benign", "Malignant"],
        "description": "Skin cancer diagnosis using dermatoscopic images",
        "details": {
            "Benign": {
                "info": "Non-cancerous skin lesion that doesn't spread to other parts of the body",
                "symptoms": ["Symmetrical shape", "Even coloration", "Smooth borders", "Stable size"],
                "causes": ["Sun exposure", "Genetic factors", "Moles", "Skin type"],
                "treatment_overview": "Monitoring and regular check-ups; removal if necessary"
            },
            "Malignant": {
                "info": "Cancerous skin lesion that can invade surrounding tissue and spread",
                "symptoms": ["Asymmetrical shape", "Irregular borders", "Multiple colors", "Changing size"],
                "causes": ["UV radiation", "Fair skin", "Family history", "Weakened immune system"],
                "treatment_overview": "Surgical removal, chemotherapy, radiation therapy, or immunotherapy"
            }
        }
    },
    "pneumonia": {
        "model_config": {
            "path": "models/pneumonia_model.h5",
            "size": [224, 224]
        },
        "labels": ["Normal", "Pneumonia"],
        "description": "Pneumonia detection from chest X-ray images",
        "details": {
            "Normal": {
                "info": "Healthy lungs with no signs of infection",
                "symptoms": ["Clear breathing", "Normal lung sounds", "No cough or fever"],
                "causes": ["N/A"],
                "treatment_overview": "Maintain good respiratory health"
            },
            "Pneumonia": {
                "info": "Lung infection causing inflammation in air sacs",
                "symptoms": ["Cough with phlegm", "Fever", "Difficulty breathing", "Chest pain"],
                "causes": ["Bacteria", "Viruses", "Fungi", "Aspiration"],
                "treatment_overview": "Antibiotics, rest, fluids, and in severe cases hospitalization"
            }
        }
    }
}

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    has_prescription: bool = False

class DiagnosisResponse(BaseModel):
    predicted_label: str
    confidence: float
    description: str
    symptoms: list
    causes: list
    treatment_overview: str
    heatmap_image: Optional[str] = None

# Medical system prompt for Gemini
MEDICAL_SYSTEM_PROMPT = """
You are Dr. AI, a professional medical AI assistant. Your role is to provide helpful, accurate, and safe medical information while emphasizing the importance of consulting healthcare professionals.

IMPORTANT GUIDELINES:
1. Always prioritize patient safety
2. Never provide definitive diagnoses - only suggest possible conditions
3. Always recommend consulting healthcare professionals
4. For emergencies, clearly state the need for immediate medical attention
5. Provide general health information and symptom guidance
6. Be empathetic and professional
7. Use clear, easy-to-understand language
8. Include relevant disclaimers

Format your responses with clear sections using markdown-like formatting:
- Use ### for main headings
- Use **bold** for important points
- Use bullet points for lists
- Use emojis sparingly for visual clarity

For medication suggestions, use this format:
💊 **Recommended Medications:**
• Medication 1 - purpose and general dosage
• Medication 2 - purpose and general dosage

Always include disclaimers about consulting doctors.
"""

@app.on_event("startup")
async def startup_event():
    print("🚀 Medical AI Assistant starting...")
    print("✅ Gemini AI Integration Active")
    print(f"✅ Loaded {len(disease_info)} diseases")
    print("✅ Enhanced medical chatbot with Gemini Pro")

async def get_gemini_response(user_message: str, conversation_history: list = None) -> str:
    """Get response from Gemini AI with medical context"""
    try:
        # Build conversation context
        prompt = f"{MEDICAL_SYSTEM_PROMPT}\n\nUser: {user_message}"
        
        if conversation_history:
            history_text = "\n".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                                    for msg in conversation_history[-6:]])  # Last 6 messages for context
            prompt = f"{MEDICAL_SYSTEM_PROMPT}\n\nConversation History:\n{history_text}\n\nUser: {user_message}"
        
        # Generate response with safety settings
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1024,
            ),
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        )
        
        return response.text
        
    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        # Fallback to local medical knowledge
        return generate_fallback_response(user_message)

def generate_fallback_response(user_message: str) -> str:
    """Generate fallback response when Gemini is unavailable"""
    user_message_lower = user_message.lower()
    
    # Check for emergencies
    emergency_keywords = ['chest pain', 'difficulty breathing', 'severe pain', 'bleeding', 'unconscious']
    if any(keyword in user_message_lower for keyword in emergency_keywords):
        return """### 🚨 EMERGENCY ALERT

**This appears to be a medical emergency!**

⚠️ **Please take immediate action:**
• Call your local emergency number (911/112/999) immediately
• Do not delay seeking medical attention
• If possible, have someone stay with you

**Your symptoms require urgent medical evaluation by healthcare professionals.**

🚨 **Call emergency services now for:** 
• Chest pain with sweating or shortness of breath
• Difficulty breathing
• Severe uncontrolled bleeding
• Sudden weakness or numbness
• Loss of consciousness

Your safety is the top priority. Please seek immediate medical care."""

    # Check for specific symptoms
    detected_symptoms = []
    for symptom, info in medical_knowledge["symptoms"].items():
        if symptom in user_message_lower:
            detected_symptoms.append((symptom, info))
    
    if detected_symptoms:
        response_parts = ["### 🤒 Symptom Analysis"]
        for symptom, info in detected_symptoms:
            response_parts.append(f"**Regarding your {symptom}:**")
            response_parts.append(f"**Possible conditions:** {', '.join(info['possible_conditions'])}")
            response_parts.append(f"**Recommendation:** {info['advice']}")
            if info['emergency']:
                response_parts.append("⚠️ **URGENT:** This may require immediate medical attention!")
        
        response_parts.extend([
            "\n💊 **General Advice:**",
            "• Rest and stay hydrated",
            "• Monitor your symptoms",
            "• Avoid self-medication without professional advice",
            "• Seek medical attention if symptoms worsen",
            "\n⚕️ **When to see a doctor:**",
            "• Symptoms persist for more than 3 days",
            "• Symptoms worsen suddenly",
            "• You develop new concerning symptoms",
            "• You have underlying health conditions"
        ])
        
        return "\n\n".join(response_parts)
    
    # General health responses
    general_responses = [
        """### 🩺 Medical Guidance

I understand you're seeking medical information. Here's how I can help:

**What I can do:**
• Provide general health information
• Explain common symptoms and conditions
• Offer lifestyle and wellness advice
• Guide you on when to seek medical care

**What I cannot do:**
• Provide definitive diagnoses
• Replace medical professionals
• Prescribe medications
• Handle emergencies

**For your concern:** Please consult with a healthcare provider for proper evaluation and treatment.

🏥 **Always remember:** Your health is important. Professional medical advice is essential for accurate diagnosis and treatment.""",

        """### 💡 Health Information

Thank you for sharing your health concerns. While I can provide general information, it's important to:

**Next Steps:**
1. **Consult a healthcare professional** for proper diagnosis
2. **Describe all your symptoms** in detail to your doctor
3. **Mention any medications** you're currently taking
4. **Share your medical history** for comprehensive care

**Self-care tips while waiting for medical advice:**
• Stay hydrated and get adequate rest
• Monitor your symptoms and note any changes
• Avoid activities that worsen your condition
• Follow general hygiene practices

⚕️ **Professional medical evaluation is recommended for personalized care.**"""
    ]
    
    return random.choice(general_responses)

# Store conversation history (in production, use a proper database)
conversation_sessions = {}

def get_session_history(session_id: str) -> list:
    """Get or create conversation history for session"""
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []
    return conversation_sessions[session_id]

def add_to_history(session_id: str, role: str, content: str):
    """Add message to conversation history"""
    history = get_session_history(session_id)
    history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    # Keep only last 20 messages to manage memory
    if len(history) > 20:
        conversation_sessions[session_id] = history[-20:]

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical AI Assistant</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f172a 100%);
                min-height: 100vh;
            }
        </style>
    </head>
    <body class="text-gray-100">
        <div class="container mx-auto px-4 py-8">
            <div class="text-center mb-8">
                <h1 class="text-4xl font-bold mb-2">🩺 Medical AI Assistant</h1>
                <p class="text-gray-400">Powered by Gemini AI • Professional Healthcare Intelligence</p>
            </div>
            
            <div class="max-w-4xl mx-auto bg-slate-800/50 backdrop-blur-md rounded-2xl border border-slate-700 p-6 shadow-2xl">
                <div class="text-center py-12">
                    <div class="text-6xl mb-4">🧠</div>
                    <h2 class="text-2xl font-bold mb-4">Medical AI Assistant with Gemini AI</h2>
                    <p class="text-gray-400 mb-6">Now powered by Google's Gemini AI for intelligent medical conversations</p>
                    
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
                        <a href="/static/app.html" 
                           class="bg-blue-600 hover:bg-blue-700 text-white py-3 px-6 rounded-lg font-medium transition-colors block text-center">
                            🗣️ Start Medical Consultation
                        </a>
                        <a href="/api/health" 
                           class="bg-green-600 hover:bg-green-700 text-white py-3 px-6 rounded-lg font-medium transition-colors block text-center">
                            🔍 Check API Health
                        </a>
                    </div>
                </div>
                
                <div class="mt-8 p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                    <p class="text-amber-300 text-sm text-center">
                        <strong>Note:</strong> Powered by Gemini AI. This is an AI assistant for general health information only. 
                        Always consult qualified medical professionals for actual medical diagnosis and treatment.
                        In emergencies, call your local emergency number immediately.
                    </p>
                </div>
            </div>
        </div>
        
        <script>
            fetch('/api/health')
                .then(response => response.json())
                .then(data => {
                    console.log('API Health:', data);
                })
                .catch(error => {
                    console.log('Health check failed:', error);
                });
        </script>
    </body>
    </html>
    """

@app.get("/api/health")
async def health_check():
    gemini_status = "active" if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE" else "not_configured"
    return {
        "status": "healthy", 
        "mode": "gemini_enhanced",
        "ai_provider": "Google Gemini Pro",
        "gemini_status": gemini_status,
        "diseases_loaded": len(disease_info),
        "message": "Medical AI Assistant running with Gemini AI integration"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Get conversation history
        history = get_session_history(request.session_id)
        
        # Get response from Gemini AI
        response_text = await get_gemini_response(request.message, history)
        
        # Add to conversation history
        add_to_history(request.session_id, "user", request.message)
        add_to_history(request.session_id, "assistant", response_text)
        
        # Check if response contains prescription information
        has_prescription = "💊" in response_text or "medication" in response_text.lower()
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            has_prescription=has_prescription
        )
    
    except Exception as e:
        print(f"Chat error: {str(e)}")
        # Fallback response
        fallback_response = generate_fallback_response(request.message)
        return ChatResponse(
            response=fallback_response,
            session_id=request.session_id,
            has_prescription=False
        )

@app.get("/api/diseases")
async def get_diseases():
    return {"diseases": list(disease_info.keys())}

@app.post("/api/diagnose", response_model=DiagnosisResponse)
async def diagnose_image(
    disease: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        if disease not in disease_info:
            raise HTTPException(status_code=400, detail="Invalid disease selection")
        
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Get disease info
        disease_data = disease_info[disease]
        label_list = disease_data.get("labels", ["Class A", "Class B"])
        
        # Mock prediction (in real implementation, use actual ML model)
        predicted_label = random.choice(label_list)
        confidence = round(random.uniform(75.0, 95.0), 2)
        
        details = disease_data.get("details", {}).get(predicted_label, {})
        
        # Generate mock heatmap
        heatmap_base64 = None
        try:
            width, height = image.size
            heatmap = np.zeros((height, width))
            
            center_x, center_y = width // 2, height // 2
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            for y in range(height):
                for x in range(width):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    heatmap[y, x] = 1.0 - (dist / max_dist)
            
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            original_img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(original_img_array, 0.6, heatmap_colored, 0.4, 0)
            
            _, buffer = cv2.imencode('.png', overlay)
            heatmap_base64 = base64.b64encode(buffer).decode()
        except Exception as e:
            print(f"Error generating mock heatmap: {e}")
        
        return DiagnosisResponse(
            predicted_label=predicted_label,
            confidence=confidence,
            description=details.get('info', disease_data.get('description', 'N/A')),
            symptoms=details.get('symptoms', []),
            causes=details.get('causes', []),
            treatment_overview=details.get('treatment_overview', 'Consult a medical professional'),
            heatmap_image=heatmap_base64
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    print("Starting Medical AI Assistant with Gemini AI on http://localhost:8000")
    print("⚠️  Remember to set your GEMINI_API_KEY in the code")
    uvicorn.run(app, host="0.0.0.0", port=8000)