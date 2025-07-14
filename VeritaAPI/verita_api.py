import mlflow.sklearn, mlflow, nltk, os, joblib, pickle, psutil, warnings, logging, multiprocessing
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings(action='ignore')
class Text(BaseModel):
    """
    Une classe basé pour sauvegarder les textes
    """
    text: str
    model_name: Optional[str] = "default"

class PredictionResponse(BaseModel):
    """
    Une classe pour les predictions et les probabilités
    Args:
        prediction: la prediction du modèle
        confidence: la confiance du modèle après prédiction
        probabilité: la justesse des résulats obtenus
    """
    prediction:int
    probability: Optional[int]=None
    prediction_label:str
    text_length:int
    model_used: str
    confidence: Optional[int]=None


class DetectorNews:

    def __init__(self, models_dir:str="/home/dona-erick/Fake_News/models"):
        """
        Initialise des répertoires des models
        Args:
            models_dir: le repertoire dans lequel se trouve les modèles
        """
        self.models_dir = Path(models_dir)
        if not self.models_dir.exists():
            raise ValueError(f" Le dossier '{models_dir}' n'existe pas")
        self.vectorizer =None
        self.models = {}
        self.load_all_models()
    # fonction pour charger tous les models et vectorizer
    def load_all_models(self):
        """
        charger tous les modèles et vectorizer 
        """
        try:
            self.load_vectorizer()

            # listes de tous les models
            models_files = list(self.models_dir.glob("*.pkl"))
            logger.info(f"Fichiers trouvés dans {self.models_dir}: {[f.name for f in models_files]}")
            for model in models_files:
                if 'vectorizer' in model.name.lower():
                    continue
                model_name = model.stem

                try:
                    self.models[model_name]= joblib.load(model)
                    logger.info(f"Modèle {model_name} chargé avec succès!")
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du modèle {model_name}: str{e}")

            if not self.models:
                logger.warning("Aucun modèle trouvé. Création d'un modèle par défaut...")

        except Exception as e:
            logger.error(f"Erreur lors de chargement des modèles: {str(e)}")

    def load_vectorizer(self, vectorizer_type: str='tfidf'):
        """
        Chargement du vectoriseur depuis son emplacement
        Args:
            vectorizer_type: le type de vectoriseur à charger 
        Returns:
            Vectoriseur chargé
        """
        try:
            vectorizer_files = list(self.models_dir.glob("*vectorizer*.pkl"))
            if not vectorizer_files:
            # chemin de vectoriseur
                vectorizer_path = self.models_dir / f'vectorizer_{vectorizer_type}.pkl'
                if vectorizer_path.exists():
                    vectorizer = joblib.load(vectorizer_path)
                    self.vectorizer = vectorizer

            if vectorizer_files:
                vectorizer_path = vectorizer_files[0]
                vectorizer = joblib.load(vectorizer_path)
                self.vectorizer=vectorizer
                logger.info(f"Le Vectorizer de type {vectorizer_type} est chargé de puis {vectorizer_path}")
                return vectorizer
            else:
                logger.error(f"Le vectorizer n'est pas chargé")
                return None
        except Exception as e:
            logger.error(f"Erreur lors du chargement du vectorizer {vectorizer_type}: {str(e)}")
            return None

    """Cette fonction valide les entrées du model et du vectorizer avant d'enclencher le 
    processus de prédiction"""

    def validate_text(self, text: str) ->str:
        """
        Cette fonction valide tout simplement le texte avant de le soumettre au model
        Args:
            text: Texte ou article à entrer
        Returns:
            Texte validé
        """

        if not text or text.strip() == "":
            raise ValueError("Le texte ne peut pas etre vide")
        else:
            logger.info("Le texte entré est validé")
        return text.strip()
    
    """Fonction pour la prediction"""

    def predict(self, text:str, model_name:str="default")-> Dict[str, Any]:
        """
        Prédit si un contenu d'article est vrai ou faux
        Args:
            text: Texte à prédire
            model_name: le nom du modèle choisi pour la prédiction
            validate_text: la fonction qui nous permet de verifier si le texte entré n'est pas vide
        Returns:
            Résultat de la prédiction
        """

        try:
            # valider le texte
            validated_text = self.validate_text(text)
            # on charge le ectoriseur 
            
            if model_name not in self.models:
                models_available = list(self.models.keys())
                logger.error(f"Le model{model_name} n'existe pas dans la liste des models disponibles {models_available}")
            model = self.models[model_name]

            if self.vectorizer is None:
                raise ValueError(f"Le vectorizer n'est pas initialisé")
            
            text_vectorized = self.vectorizer.transform([validated_text])
            prediction = model.predict(text_vectorized)[0]

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(text_vectorized)[0].tolist()
                confidence = max(probability)
            else:
                probability = None
                confidence= None

            return {
                "prediction": int(prediction),
                "prediction_label": "Fake News" if prediction == 0 else "Real News",
                "probability": probability,
                "text_length": len(validated_text),
                "confidence": confidence,
                "model_used": model_name
            } 

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")
        

""" Mise en Place API"""

app = FastAPI(title="VeritaAPI",
              description="API de détection des fausses nouvelles et des vraies nouvelles",
              version="0.1.0",
              debug=True)

# initialisé le detector
detector = DetectorNews("/home/dona-erick/Fake_News/models")

# route 1
@app.get("/")

def home():
    """
    Page d'accueil de l'API
    """
    return {
        "message": "Bienvenue sur VeritaAI - Détecteur de Fake News",
        "description": "Plateforme de détection et vérification d'informations",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict/": "Prédire si un article est une fake news",
            "GET /models/": "Lister les modèles disponibles",
            "GET /health/": "Vérifier le statut de l'API"
        }
    }


@app.post('/predict/', response_model= PredictionResponse)

def detecteur_fake(input_data: Text):

    try:
        result = detector.predict(input_data.text, model_name=input_data.model_name)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Erreur dans l'endpoint predict: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/models/')

def list_models():
    """
    Liste des modèles disponibles
    """
    return {
        "Models disponibles": list(detector.models.keys()),
        "Nombre de Modèles": len(detector.models),
        "Vectorizer": detector.vectorizer is not None,
    }

@app.get("/health/")
def health_check():
    """
    Vérifie le statut de l'API
    """
    return {
        "status": "healthy",
        "models_loaded": len(detector.models),
        "vectorizer_status": "loaded" if detector.vectorizer else "not_loaded",
        "memory_usage": f"{psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB"
    }

@app.post('/batch_predict/')
def batch_predict(texts: list[str], model_name: str = "default"):
    try:
        results = []
        for i, text in enumerate(texts):
            try:
                result = detector.predict(text, model_name)
                result['text_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'text_index': i,
                    'error': str(e),
                    'prediction': None,
                    'prediction_label': 'Error'
                })
        
        return {
            "total_texts": len(texts),
            "successful_predictions": len([r for r in results if 'error' not in r]),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        host="127.0.0.1",
        host="127.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )