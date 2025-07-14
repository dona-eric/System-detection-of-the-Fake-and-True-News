from visualisation import DataVisualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, sys, os, re, joblib, xgboost, logging, psutil, multiprocessing
from pathlib import Path
from typing import Union, Tuple, Dict, Any
import nltk, mlflow, mlflow.sklearn
mlflow.set_tracking_uri("http://127.0.0.1:5000")
from mlflow.models import infer_signature
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer, LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score,learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, auc, roc_curve,confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import resample
warnings.filterwarnings("ignore")
# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
print(f"Nombre de CPUs disponibles: {multiprocessing.cpu_count()}")
print(f"Mémoire disponible: {psutil.virtual_memory().available / 1024**3:.2f} GB")
# Télécharger les ressources NLTK nécessaires
def download_nltk_resources():
    """Télécharge les ressources NLTK nécessaires"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.download('punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Téléchargement des ressources NLTK...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('omw-1.4', quiet=True)

class DataProcessor:

    """Classe pour le traitement des données de détection de fausses nouvelles"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialise le processeur de données
        
        Args:
            models_dir (str): Répertoire pour sauvegarder les modèles
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        # Initialiser les outils de traitement
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        
        download_nltk_resources()
        logger.info("DataProcessor initialisé avec succès")
    
    
    def clean_text(self, text: str):
        if pd.isnull(text):
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def preprocess_text(self, text: str, use_stemming: bool = False):
        """
        Prétraite un texte avec tokenisation, lemmatisation et suppression des stopwords
        
        Args:
            text (str): Texte à prétraiter
            use_stemming (bool): Utiliser le stemming au lieu de la lemmatisation
            
        Returns:
            str: Texte prétraité
        """
        if pd.isnull(text):
            return ""
        text = self.clean_text(text)
        if not text:
            return ""
        ## Tokenisation et pretraitement
        try:
            # Tokenisation
            tokens = word_tokenize(text)
            # Supprimer les stopwords et les mots trop courts
            tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
            # Lemmatisation ou stemming
            if use_stemming:
                tokens = [self.stemmer.stem(word) for word in tokens]
            else:
                tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            return ' '.join(tokens)
        except Exception as e:
            logger.warning(f"Erreur lors du prétraitement du texte : {str(e)}")
            return ""
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text', 
                           use_stemming: bool = False):
        """
        Prétraite une colonne de texte dans un DataFrame
        Args:
            df (pd.DataFrame): DataFrame à traiter
            text_column (str): Nom de la colonne contenant le texte
            use_stemming (bool): Utiliser le stemming au lieu de la lemmatisation
        Returns:
            pd.DataFrame: DataFrame avec texte prétraité
        """
        if text_column not in df.columns:
            raise ValueError(f"La colonne '{text_column}' n'existe pas dans le DataFrame")
        
        logger.info(f"Prétraitement de la colonne '{text_column}'...")
        # Créer une copie du DataFrame
        df_processed = df.copy()
        # Appliquer le prétraitement
        df_processed[f'{text_column}_processed'] = df_processed[text_column].apply(
            lambda x: self.preprocess_text(x, use_stemming)
        )
        # Supprimer les lignes vides après prétraitement
        df_processed = df_processed[df_processed[f'{text_column}_processed'].str.len() > 0]
        
        logger.info(f"Prétraitement terminé. {len(df_processed)} lignes conservées")
        
        return df_processed
    
    def vectorize_text(self, X_train: pd.Series, X_test: pd.Series, 
                      vectorizer_type: str = 'tfidf', 
                      max_features: int = 10000,
                      ngram_range: Tuple[int, int] = (1, 2),
                      save_vectorizer: bool = True) -> Tuple[np.ndarray, np.ndarray, Union[TfidfVectorizer, CountVectorizer]]:
        """
        Vectorise les textes d'entraînement et de test
        
        Args:
            X_train (pd.Series): Textes d'entraînement
            X_test (pd.Series): Textes de test
            vectorizer_type (str): Type de vectoriseur ('tfidf' ou 'count')
            max_features (int): Nombre maximum de features
            ngram_range (Tuple[int, int]): Plage de n-grammes
            save_vectorizer (bool): Sauvegarder le vectoriseur
            
        Returns:
            Tuple: (X_train_vectorized, X_test_vectorized, vectorizer)
            
        Raises:
            ValueError: Si le type de vectoriseur n'est pas supporté
        """
        logger.info(f"Vectorisation avec {vectorizer_type}...")
        
        # Initialiser le vectoriseur
        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95
            )
        elif vectorizer_type == 'count':
            vectorizer = CountVectorizer(
                stop_words='english',
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95
            )
        else:
            raise ValueError("vectorizer_type doit être 'tfidf' ou 'count'")
        
        try:
            # Vectorisation
            X_train_vectorized = vectorizer.fit_transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)
            
            # Sauvegarder le vectoriseur
            if save_vectorizer:
                vectorizer_path = self.models_dir / f'vectorizer_{vectorizer_type}.pkl'
                joblib.dump(vectorizer, vectorizer_path)
                logger.info(f"Vectoriseur sauvegardé : {vectorizer_path}")
            self.vectorizer = vectorizer
            logger.info(f"Vectorisation terminée - Shape train: {X_train_vectorized.shape}, Shape test: {X_test_vectorized.shape}")
            return X_train_vectorized, X_test_vectorized, vectorizer
            
        except Exception as e:
            logger.error(f"Erreur lors de la vectorisation : {str(e)}")
            raise
    
    def load_vectorizer(self, vectorizer_type: str = 'tfidf') -> Union[TfidfVectorizer, CountVectorizer]:
        """
        Charge un vectoriseur sauvegardé
        Args:
            vectorizer_type (str): Type de vectoriseur à charger
            
        Returns:
            Union[TfidfVectorizer, CountVectorizer]: Vectoriseur chargé
        """
        vectorizer_path = self.models_dir / f'vectorizer_{vectorizer_type}.pkl'
        
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectoriseur non trouvé : {vectorizer_path}")
        
        try:
            vectorizer = joblib.load(vectorizer_path)
            self.vectorizer = vectorizer
            logger.info(f"Vectoriseur chargé : {vectorizer_path}")
            return vectorizer
        except Exception as e:
            logger.error(f"Erreur lors du chargement du vectoriseur : {str(e)}")
            raise
    
    def prepare_data_for_training(self, df: pd.DataFrame, 
                                 text_column: str = 'text',
                                 label_column: str = 'label',
                                 test_size: float = 0.2,
                                 random_state: int = 42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Prépare les données pour l'entraînement
        
        Args:
            df (pd.DataFrame): DataFrame avec les données
            text_column (str): Nom de la colonne de texte
            label_column (str): Nom de la colonne de labels
            test_size (float): Proportion des données de test
            random_state (int): Seed pour la reproductibilité
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Préparation des données pour l'entraînement...")
        
        # Vérifier les colonnes
        if text_column not in df.columns:
            raise ValueError(f"Colonne '{text_column}' non trouvée")
        if label_column not in df.columns:
            raise ValueError(f"Colonne '{label_column}' non trouvée")
        
        # Prétraiter les données
        df_processed = self.preprocess_dataframe(df, text_column)
        
        # Préparer X et y
        X = df_processed[f'{text_column}_processed']
        y = df_processed[label_column]
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Division des données - Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Obtient des informations sur le dataset
        
        Args:
            df (pd.DataFrame): DataFrame à analyser
            
        Returns:
            dict: Informations sur les données
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Informations sur les labels si présents
        if 'label' in df.columns:
            info['label_distribution'] = df['label'].value_counts().to_dict()      
        return info
    # fonction pour entrainement du modèle
    def train_model(self, X_train, y_train):
        """
        Entraîne plusieurs modèles de machine learning
        
        Args:
            X_train: Données d'entraînement vectorisées
            y_train: Labels d'entraînement
            
        Yields:
            Tuple: (nom_modele, modele_entrainé)
        """
        models_and_params = {
            "LogisticRegression": (
                LogisticRegression(max_iter=1000),
                {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'saga']
                }
            ),
            "LinearSVC": (
                LinearSVC(),
                {
                    'C': [0.01, 0.1, 1, 10],         
                    'max_iter': [1000, 3000, 5000],
                    'tol': [1e-4, 1e-3],   
                    'loss': ['hinge', 'squared_hinge']
                }
            ),
            "MultinomialNB": (
                MultinomialNB(),
                {
                    'alpha': [0.1, 0.5, 1.0],
                    'fit_prior': [True, False]
                }
            ),
            "RandomForest": (
                RandomForestClassifier(random_state=42),
                {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'bootstrap': [True]
                }
            ),
            "XGBoost": (
                XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.7, 1.0],
                }
            ),

            "DecisionTree": (
                DecisionTreeClassifier(random_state=42),
                {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'criterion':['gini']
                }
            ),
            "GradientBoosting": (
                GradientBoostingClassifier(),
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5],
                'subsample': [0.8, 1.0]
            }
        )
    }

        for name, (model, params) in models_and_params.items():
            logger.info(f"Entraînement et Validation croisée du {name}...")
            try:
                grid = GridSearchCV(model, param_grid=params, cv=5, scoring='accuracy',n_jobs=-1, verbose=1 )
                grid.fit(X_train, y_train)
               # Récupérer le meilleur modèle
                best_model = grid.best_estimator_
                # Validation croisée sur le meilleur modèle
                cv_scores = cross_val_score(
                    best_model, X_train, y_train, 
                    cv=5, scoring='f1_weighted'
                )
                
                logger.info(f"{name} entraîné avec succès.")
                logger.info(f"Meilleur score CV: {grid.best_score_:.4f}")
                logger.info(f"Score CV moyen: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                yield name, best_model, grid.best_params_, cv_scores
                
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de {name}: {str(e)}")
                continue

    def plot_learning_curves(self, model, X_train, y_train, model_name: str):
        """
        Génère et sauvegarde les courbes d'apprentissage pour détecter l'overfitting
        
        Args:
            model: Modèle à analyser
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            model_name: Nom du modèle
            
        Returns:
            str: Chemin vers le fichier de la courbe sauvegardée
        """
        try:
            # Créer le répertoire pour les graphiques s'il n'existe pas
            plots_dir = Path("mlflow_plots")
            plots_dir.mkdir(exist_ok=True)
            
            # Calculer les courbes d'apprentissage
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=train_sizes,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                random_state=42
            )
            
            # Calculer les moyennes et écarts-types
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Créer le graphique
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Score d\'entraînement')
            plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Score de validation')
            plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            plt.xlabel('Nombre d\'échantillons d\'entraînement')
            plt.ylabel('Score F1')
            plt.title(f'Courbes d\'apprentissage - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Analyser l'overfitting
            final_train_score = train_mean[-1]
            final_val_score = val_mean[-1]
            gap = final_train_score - final_val_score
            
            # Ajouter une annotation sur l'overfitting
            if gap > 0.1:
                plt.text(0.6, 0.1, f'Overfitting détecté\nÉcart: {gap:.3f}', 
                        transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            elif gap > 0.05:
                plt.text(0.6, 0.1, f'Overfitting modéré\nÉcart: {gap:.3f}', 
                        transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
            else:
                plt.text(0.6, 0.1, f'Pas d\'overfitting\nÉcart: {gap:.3f}', 
                        transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7))
            
            plt.tight_layout()
            
            # Sauvegarder
            plot_path = plots_dir / f'learning_curve_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération des courbes d'apprentissage pour {model_name}: {str(e)}")
            return None

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str) -> str:
        """
        Génère et sauvegarde la matrice de confusion
        """
        try:
            plots_dir = Path("mlflow_plots")
            plots_dir.mkdir(exist_ok=True)
            
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            
            # Créer un heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fake', 'Real'], 
                    yticklabels=['Fake', 'Real'])
            plt.title(f'Matrice de Confusion - {model_name}')
            plt.xlabel('Prédictions')
            plt.ylabel('Valeurs réelles')
            
            plot_path = plots_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la matrice de confusion pour {model_name}: {str(e)}")
            return None

    def plot_roc_curve(self, model, X_test, y_test, model_name: str):
        """
        Génère et sauvegarde la courbe ROC
        """
        try:
            plots_dir = Path("mlflow_plots")
            plots_dir.mkdir(exist_ok=True)
            
            # Calculer les probabilités de prédiction
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                logger.warning(f"Impossible de générer la courbe ROC pour {model_name}")
                return None
            
            # Calculer la courbe ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'Courbe ROC (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taux de Faux Positifs')
            plt.ylabel('Taux de Vrais Positifs')
            plt.title(f'Courbe ROC - {model_name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            plot_path = plots_dir / f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path), roc_auc
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la courbe ROC pour {model_name}: {str(e)}")
            return None, None

    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict[str, Any]:
        """
        Évalue les performances du modèle avec visualisations
        
        Args:
            model: Modèle à évaluer
            X_test: Données de test
            y_test: Labels de test
            model_name: Nom du modèle
            
        Returns:
            Dict: Métriques de performance et chemins des graphiques
        """
        try:
            y_pred = model.predict(X_test)
            
            # Calculer les métriques
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # Générer les visualisations
            cm_path = self.plot_confusion_matrix(y_test, y_pred, model_name)
            roc_result = self.plot_roc_curve(model, X_test, y_test, model_name)
            
            if roc_result and len(roc_result) == 2:
                roc_path, roc_auc = roc_result
            else:
                roc_path, roc_auc = None, None
            
            logger.info("Rapport de classification :")
            logger.info(f"\n{classification_report(y_test, y_pred)}")
            logger.info(f"\nMétriques de performance:")
            logger.info(f"Précision (Accuracy): {accuracy:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info(f"Précision (Precision): {precision:.4f}")
            logger.info(f"Rappel (Recall): {recall:.4f}")
            if roc_auc:
                logger.info(f"AUC-ROC: {roc_auc:.4f}")
            
            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'confusion_matrix_path': cm_path,
                'roc_curve_path': roc_path
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation du modèle: {str(e)}")
            raise

    def train_and_evaluate_models(self, df: pd.DataFrame, 
                                text_column: str = 'cleaned_text',
                                label_column: str = 'label',
                                vectorizer_type: str = 'tfidf',
                                test_size: float = 0.2,
                                random_state: int = 42,
                                experiment_name: str = "Verita Experiment") -> pd.DataFrame:
        """
        Entraîne et évalue tous les modèles avec suivi MLflow complet
        
        Args:
            df (pd.DataFrame): DataFrame avec les données
            text_column (str): Nom de la colonne de texte
            label_column (str): Nom de la colonne de labels
            vectorizer_type (str): Type de vectoriseur
            test_size (float): Proportion des données de test
            random_state (int): Seed pour la reproductibilité
            experiment_name (str): Nom de l'expérience MLflow
            
        Returns:
            pd.DataFrame: Résumé des performances
        """
        logger.info("Démarrage de l'entraînement et évaluation des modèles...")
        
        # Préparer les données
        X_train, X_test, y_train, y_test = self.prepare_data_for_training(
            df, text_column, label_column, test_size, random_state
        )
        
        # Vectoriser les données
        X_train_vectorized, X_test_vectorized, vectorizer = self.vectorize_text(
            X_train, X_test, vectorizer_type
        )
        
        # Configurer MLflow
        mlflow.set_experiment(experiment_name)
        
        # Entraîner et évaluer les modèles
        results = {}
        
        for name, model, best_params, cv_scores in self.train_model(X_train_vectorized, y_train):
            with mlflow.start_run(run_name=name):
                try:
                    # Enregistrer les paramètres
                    mlflow.log_param("model_type", name)
                    mlflow.log_param("vectorizer_type", vectorizer_type)
                    mlflow.log_param("test_size", test_size)
                    mlflow.log_param("random_state", random_state)
                    mlflow.log_param("max_features", vectorizer.max_features)
                    mlflow.log_param("ngram_range", str(vectorizer.ngram_range))
                    
                    # Enregistrer les meilleurs paramètres
                    for param_name, param_value in best_params.items():
                        mlflow.log_param(f"best_{param_name}", param_value)
                    
                    # Enregistrer les scores de validation croisée
                    mlflow.log_metric("cv_score_mean", cv_scores.mean())
                    mlflow.log_metric("cv_score_std", cv_scores.std())
                    for i, score in enumerate(cv_scores):
                        mlflow.log_metric(f"cv_score_fold_{i+1}", score)
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Évaluation du modèle {name}")
                    logger.info(f"{'='*60}")
                    
                    # Générer les courbes d'apprentissage
                    learning_curve_path = self.plot_learning_curves(
                        model, X_train_vectorized, y_train, name
                    )
                    
                    # Évaluer le modèle
                    metrics = self.evaluate_model(model, X_test_vectorized, y_test, name)
                    results[name] = metrics
                    
                    # Enregistrer les métriques dans MLflow
                    mlflow.log_metric("accuracy", metrics['accuracy'])
                    mlflow.log_metric("f1_score", metrics['f1_score'])
                    mlflow.log_metric("precision", metrics['precision'])
                    mlflow.log_metric("recall", metrics['recall'])
                    if metrics['roc_auc']:
                        mlflow.log_metric("roc_auc", metrics['roc_auc'])
                    
                    # Enregistrer les visualisations dans MLflow
                    if learning_curve_path:
                        mlflow.log_artifact(learning_curve_path, "plots")
                    if metrics['confusion_matrix_path']:
                        mlflow.log_artifact(metrics['confusion_matrix_path'], "plots")
                    if metrics['roc_curve_path']:
                        mlflow.log_artifact(metrics['roc_curve_path'], "plots")
                    
                    # Enregistrer le modèle
                    try:
                        mlflow.sklearn.log_model(
                            model, 
                            "model",
                            input_example=X_test_vectorized[:1].toarray() if hasattr(X_test_vectorized, 'toarray') else X_test_vectorized[:1],
                            signature=infer_signature(
                                X_test_vectorized.toarray() if hasattr(X_test_vectorized, 'toarray') else X_test_vectorized, 
                                metrics['predictions']
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Erreur lors de l'enregistrement du modèle avec signature: {e}")
                        mlflow.sklearn.log_model(model, "model")
                    
                    # Sauvegarder le modèle localement
                    model_path = self.models_dir / f'model_{name.lower().replace(" ", "_")}.pkl'
                    joblib.dump(model, model_path)
                    logger.info(f"Modèle sauvegardé : {model_path}")
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du modèle {name}: {str(e)}")
                    continue
        
        # Afficher un résumé des résultats
        if results:
            logger.info(f"\n{'='*60}")
            logger.info("RÉSUMÉ DES PERFORMANCES")
            logger.info(f"{'='*60}")
            
            # Créer le DataFrame des résultats
            results_df = pd.DataFrame(results).T
            # Supprimer les colonnes non numériques pour l'affichage
            display_cols = ['accuracy', 'f1_score', 'precision', 'recall']
            if 'roc_auc' in results_df.columns:
                display_cols.append('roc_auc')
            
            display_df = results_df[display_cols].round(4)
            logger.info(f"\n{display_df.to_string()}")
            
            # Trouver le meilleur modèle
            best_model = display_df['f1_score'].idxmax()
            logger.info(f"\nMeilleur modèle selon le F1-score: {best_model}")
            logger.info(f"F1-score: {display_df.loc[best_model, 'f1_score']:.4f}")
            
            # Créer un graphique comparatif des performances
            self.plot_model_comparaison(display_df)
            
            return display_df
        else:
            logger.error("Aucun modèle n'a été entraîné avec succès")
            return pd.DataFrame()

    def plot_model_comparaison(self, results_df: pd.DataFrame) -> str:
        """
        Génère un graphique comparatif des performances des modèles
        """
        try:
            plots_dir = Path("mlflow_plots")
            plots_dir.mkdir(exist_ok=True)
            
            # Créer un graphique comparatif
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Comparaison des Performances des Modèles', fontsize=16)
            
            metrics = ['accuracy', 'f1_score', 'precision', 'recall']
            colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']
            
            for i, (metric, color) in enumerate(zip(metrics, colors)):
                ax = axes[i//2, i%2]
                results_df[metric].plot(kind='bar', ax=ax, color=color)
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = plots_dir / 'model_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique comparatif: {str(e)}")
            return None

    def predict_new_text(self, model, text: str) -> Dict[str, Any]:
        """
        Prédit si un nouveau texte est une fake news
        Args:
            model: Modèle entraîné
            text (str): Texte à prédire
        Returns:
            Dict: Résultat de la prédiction
        """
        if self.vectorizer is None:
            raise ValueError("Vectoriseur non initialisé. Veuillez d'abord entraîner un modèle ou charger un vectoriseur.")
        
        try:
            processed_text = self.preprocess_text(text)
            vectorized_text = self.vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)[0]
            
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(vectorized_text)[0]
                confidence = max(probability)
            else:
                probability = None
                confidence = None
            
            return {
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence,
                'processed_text': processed_text
            }
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise

# Fonctions utilitaires 
def load_data(file_path: str):
    """Fonction pour charger les données"""
    processor = DataProcessor()
    return processor.load_data(file_path)

def preprocess_text(text: str):
    """Fonction pour le prétraitement"""
    processor = DataProcessor()
    return processor.preprocess_text(text)

def vectorize_text(X_train, X_test, vectorizer_type='tfidf'):
    """Fonction pour la vectorisation"""
    processor = DataProcessor()
    return processor.vectorize_text(X_train, X_test, vectorizer_type)

if __name__ == "__main__":
    # Initialiser le processeur
    processor = DataProcessor()
    
    # Charger les données
    try:
        df = processor.load_data("/home/dona-erick/Fake_News/Data_cleaned/data_cleaned.csv")
        # Obtenir des informations sur les données
        info = processor.get_data_info(df)
        logger.info(f"Informations sur les données: {info}")
        
        # Entraîner et évaluer les modèles
        results = processor.train_and_evaluate_models(df)
        logger.info("Analyse terminée avec succès!")
        
        # la courbe d'apprentissage 
        X_train, X_test,y_train, y_test = processor.prepare_data_for_training(df=df)
        curve = processor.plot_learning_curves(X_train, y_train)
        logger.info("Courbe d'apprentissage avec suivi du model")
        # matrice de confusion
        matrice = processor.plot_confusion_matrix(df)
        logger.info("Matrice de confusion")
        # comparaison entre les models
        comparaison = processor.plot_model_comparaison(df)
        logger.info("Comparaison entre les modèles")
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}")

if __name__ == "__main__":
    # Initialiser le processeur
    processor = DataProcessor()
    data = DataVisualizer()
    # Charger les données
    try:
        df = data.load_data("/home/dona-erick/Fake_News/Data_cleaned/data_cleaned.csv")
        # Obtenir des informations sur les données
        info = processor.get_data_info(df)
        logger.info(f"Informations sur les données: {info}")
        
        # Entraîner et évaluer les modèles
        results = processor.train_and_evaluate_models(df)
        logger.info("Analyse terminée avec succès!")
        
        # la courbe d'apprentissage 
        X_train, X_test,y_train, y_test = processor.prepare_data_for_training(df=df)
        curve = processor.plot_learning_curves(X_train, y_train)
        logger.info("Courbe d'apprentissage avec suivi du model")
        # matrice de confusion
        matrice = processor.plot_confusion_matrix(df)
        logger.info("Matrice de confusion")
        # comparaison entre les models
        comparaison = processor.plot_model_comparaison(df)
        logger.info("Comparaison entre les modèles")
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}")