# 📧 Smart Email Classifier (Spam Detection + AI Response Generator)

This project is a **Spam Email Classifier** with a modern web interface, built using **Machine Learning (scikit-learn)** and **Flask**.  
It not only detects whether an email is **Spam** or **Not Spam**, but also uses **Google Gemini API** to generate professional email replies for safe emails.

---

## ✨ Features
- 🔍 **Spam Detection** using a trained Naive Bayes model with TF-IDF vectorization.
- 📊 **Confidence Score** shown for every prediction.
- 🤖 **AI-Powered Responses** for non-spam emails via **Google Gemini API**.
- 🎨 **Modern Web UI** with dark/light mode and interactive elements.
- 📂 **Reusable ML Pipeline** for retraining and saving models.

---

## 🛠️ Tech Stack
- **Backend:** Flask, scikit-learn, joblib
- **Frontend:** HTML5, CSS3, JavaScript (Jinja2 templating)
- **ML Pipeline:** TF-IDF Vectorizer + Multinomial Naive Bayes
- **NLP Tools:** NLTK (stopwords, lemmatization)
- **AI Integration:** Google Gemini API (`google-generativeai`)
- **Deployment:** Gunicorn (for production readiness)

---

## 📂 Project Structure
```
templates/
 └── index.html         # Web UI (dark/light mode, results, response)
app.py                  # Flask app (loads model, serves predictions)
spam_classifier.py      # ML training pipeline (preprocessing + training + saving models)
spam_dataset.csv        # Dataset for training (sample dataset included)
spam_model.pkl          # Trained classifier (saved model)
tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
requirements.txt        # Python dependencies
config.env              # Local environment variables (ignored in Git)
config.env.example      # Example environment config
README.md               # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Aneesh241/spam-classifier.git
cd spam-classifier
```

### 2. Create & activate a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy the example file and set your own values:
```bash
cp config.env.example config.env
```

Edit `config.env`:
```ini
# API Keys
GOOGLE_API_KEY=your_google_api_key_here

# Flask Config
SECRET_KEY=your_secret_key_here
FLASK_DEBUG=False
PORT=5000
```

⚠️ **Note:** `config.env` is ignored by Git to protect secrets.

---

## ▶️ Running the Application

```bash
python app.py
```

Flask will start locally on:
```
http://127.0.0.1:5000/
```

Paste an email message into the text area → see whether it’s **Spam** or **Not Spam**.  
If **Not Spam**, the app will also suggest an **AI-generated email reply**.

---

## 📊 Training / Retraining the Model
If you want to retrain with a new dataset:

```bash
python spam_classifier.py
```

This will:
1. Load and clean `spam_dataset.csv`.
2. Balance dataset (upsample/downsample).
3. Train with **TF-IDF + Naive Bayes**.
4. Perform **GridSearchCV** hyperparameter tuning.
5. Save `spam_model.pkl` and `tfidf_vectorizer.pkl`.

---

## 🌐 Deployment
For production, run with Gunicorn:
```bash
gunicorn app:app
```

Can be deployed on:
- **Render**
- **Railway**
- **Heroku**
- **Dockerized containers**

---

## 🔒 Security Notes
- Never commit your real `config.env` or API keys.
- Rotate your **Google API key** if it was exposed.
- Use `.gitignore` (already included) to keep secrets safe.

---

## 📜 License
MIT License – You are free to use, modify, and distribute this project for educational and personal purposes.

---

## 👨‍💻 Author
**Aneesh Sagar Reddy**  
B.Tech CSE (AI & Engineering), Amrita School of Engineering  
Passionate about **Cybersecurity, AI, and Web Development**.
