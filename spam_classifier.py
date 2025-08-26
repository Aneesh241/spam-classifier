import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    logger.warning(f"NLTK resources could not be downloaded: {e}")
    STOPWORDS = set()
    class DummyLemmatizer:
        def lemmatize(self, word, pos='v'):
            return word
    lemmatizer = DummyLemmatizer()

def load_data(file_path="spam_dataset.csv"):
    """Load and prepare the dataset."""
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        logger.info(f"Loaded dataset with {len(df)} rows")
        
        # Keep only necessary columns and rename them
        df = df[['Category', 'Message']].rename(columns={
            'Category': 'label',
            'Message': 'text'
        })
        
        # Normalize labels
        df['label'] = df['label'].str.lower().str.strip()
        df['label'] = df['label'].map({'ham': 0, 'spam': 1, 'no': 0, 'yes': 1})
        df['label'] = df['label'].fillna(0).astype(int)
        
        logger.info(f"Label distribution before balancing: {df['label'].value_counts().to_dict()}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_text(text):
    """Advanced text preprocessing."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stopwords
    tokens = [word for word in text.split() if word.isalnum() and word not in STOPWORDS]
    
    # Lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def balance_data(df, method='upsample'):
    """Balance the dataset using various methods."""
    # Separate majority and minority classes
    ham = df[df.label == 0]
    spam = df[df.label == 1]
    
    if method == 'upsample':
        # Upsample minority class
        if len(spam) < len(ham):
            spam_upsampled = resample(spam, 
                                      replace=True,
                                      n_samples=len(ham), 
                                      random_state=42)
            df_balanced = pd.concat([ham, spam_upsampled])
        else:
            ham_upsampled = resample(ham, 
                                     replace=True,
                                     n_samples=len(spam), 
                                     random_state=42)
            df_balanced = pd.concat([ham_upsampled, spam])
    
    elif method == 'downsample':
        # Downsample majority class
        if len(ham) > len(spam):
            ham_downsampled = resample(ham, 
                                       replace=False,
                                       n_samples=len(spam), 
                                       random_state=42)
            df_balanced = pd.concat([ham_downsampled, spam])
        else:
            spam_downsampled = resample(spam, 
                                        replace=False,
                                        n_samples=len(ham), 
                                        random_state=42)
            df_balanced = pd.concat([ham, spam_downsampled])
    
    logger.info(f"Label distribution after balancing: {df_balanced['label'].value_counts().to_dict()}")
    
    return df_balanced

def train_model(save_path='.'):
    """Train and save the spam classification model."""
    try:
        # Load and prepare data
        df = load_data()
        logger.info("Cleaning text data...")
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Balance the dataset
        df = balance_data(df, method='upsample')
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )
        
        # Create a pipeline for preprocessing and classification
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ])
        
        # Hyperparameter tuning with GridSearchCV
        logger.info("Performing hyperparameter tuning...")
        param_grid = {
            'tfidf__max_features': [3000, 5000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__alpha': [0.1, 0.3, 0.5, 1.0]
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification Report:\n{report}")
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, df['cleaned_text'], df['label'], cv=5, scoring='f1')
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean CV F1 score: {np.mean(cv_scores):.4f}")
        
        # Save the model
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, 'spam_model.pkl')
        tfidf_path = os.path.join(save_path, 'tfidf_vectorizer.pkl')
        
        # Extract TF-IDF vectorizer and classifier from pipeline
        tfidf = best_model.named_steps['tfidf']
        classifier = best_model.named_steps['classifier']
        
        joblib.dump(classifier, model_path)
        joblib.dump(tfidf, tfidf_path)
        
        logger.info(f"Model saved successfully at {model_path}")
        logger.info(f"TF-IDF vectorizer saved at {tfidf_path}")
        
        return best_model
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

if __name__ == '__main__':
    train_model()
