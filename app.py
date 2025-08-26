from flask import Flask, request, render_template, flash
import joblib
import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from config.env file
load_dotenv('config.env')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))

# Load model and vectorizer
try:
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    logger.info("ML models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    model = None
    vectorizer = None

# Gemini setup
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini = genai.GenerativeModel("gemini-2.0-flash")
    logger.info("Gemini API configured successfully")
else:
    logger.warning("No Gemini API key found - response generation will be disabled")
    model_gemini = None

def clean_text(text):
    """Clean and normalize input text for spam classification."""
    if not text:
        return ""
    text = str(text).lower()
    text = ' '.join(word for word in text.split() if word.isalnum())
    return text

def generate_email_response(email_text):
    """Generate a response to an email using the Gemini API."""
    if not GOOGLE_API_KEY or not model_gemini:
        return "API key not configured. Please set GOOGLE_API_KEY in environment variables."
    
    prompt = (
        f"You are an AI assistant. Read the following email:\n\n"
        f"\"{email_text}\"\n\n"
        f"Write a clear, polite, and professional response that addresses the main points of the email. "
        f"Keep it concise and friendly."
    )
    
    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating response with Gemini: {e}")
        return f"Error generating response: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route handler for the spam classifier application."""
    result = None
    ai_response = None
    confidence_score = None

    if request.method == 'POST':
        email_text = request.form.get('email', '').strip()
        
        if not email_text:
            flash("Please enter email content to analyze")
            return render_template('index.html')
        
        if not model or not vectorizer:
            flash("ML models not loaded. Please check server logs.")
            return render_template('index.html')
            
        try:
            # Classify email
            cleaned = clean_text(email_text)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            
            # Get probability scores
            proba = model.predict_proba(vectorized)[0]
            confidence_score = round(proba[prediction] * 100, 2)
            
            result = "Spam" if prediction == 1 else "Not Spam"
            logger.info(f"Classification result: {result} with confidence {confidence_score}%")

            # Generate response for non-spam emails
            if result == "Not Spam":
                ai_response = generate_email_response(email_text)
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            flash(f"Error during prediction: {str(e)}")
            
    return render_template('index.html', 
                          result=result, 
                          ai_response=ai_response,
                          confidence_score=confidence_score)

if __name__ == '__main__':
    # Get configuration from environment
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    
    # Check if models are loaded
    if not model or not vectorizer:
        logger.warning("Warning: ML models could not be loaded. Application may not function correctly.")
    
    # Run the Flask app
    logger.info(f"Starting application on port {port}, debug mode: {debug_mode}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
