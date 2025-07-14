import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mlflow
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class DataVisualizer:
    """
    Classe pour l'analyse exploratoire des données et la visualisation
    """
    
    def __init__(self):
        self.plots_dir = Path("mlflow_plots")
        self.plots_dir.mkdir(exist_ok=True)

    def load_data(self, file_path: Union[str, Path]):
        """
        Charge les données depuis un fichier CSV

        Args:
            file_path (Union[str, Path]): Chemin vers le fichier CSV
            
        Returns:
            pd.DataFrame: DataFrame avec les données chargées
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
        try:
            logger.info(f"Chargement des données depuis {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Données chargées avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes")
            return df
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du fichier : {str(e)}")
        
        """Pour analyser et visualiser les données"""
    def analyze_and_visualize_data(self, df: pd.DataFrame, 
                                  text_column: str = 'cleaned_text',
                                  label_column: str = 'label') -> str:
        """
        Analyse et visualise les données avant l'entraînement
        
        Args:
            df: DataFrame avec les données
            text_column: Nom de la colonne de texte
            label_column: Nom de la colonne de labels
            
        Returns:
            str: Chemin vers le fichier de visualisation
        """
        try:
            # Nettoyer les données
            df_clean = df.dropna(subset=[text_column]).copy()
            
            # Créer une figure avec plusieurs sous-graphiques
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Analyse Exploratoire des Données', fontsize=16)
            
            # 1. Distribution des labels
            label_counts = df_clean[label_column].value_counts().to_dict()
            axes[0, 0].pie(label_counts.values, 
                          labels=['Fake News', 'Real News'], 
                          autopct='%1.1f%%',
                          colors=['lightcoral', 'lightgreen'])
            axes[0, 0].set_title('Distribution des Labels')
            
            # 2. Longueur des textes
            df_clean['text_length'] = df_clean[text_column].str.len()
            axes[0, 1].hist(df_clean['text_length'], bins=50, alpha=0.7, color='skyblue')
            axes[0, 1].set_title('Distribution de la Longueur des Textes')
            axes[0, 1].set_xlabel('Longueur (caractères)')
            axes[0, 1].set_ylabel('Fréquence')
            
            # 3. Nombre de mots
            df_clean['word_count'] = df_clean[text_column].str.split().str.len()
            axes[1, 0].hist(df_clean['word_count'], bins=50, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Distribution du Nombre de Mots')
            axes[1, 0].set_xlabel('Nombre de mots')
            axes[1, 0].set_ylabel('Fréquence')
            
            # 4. Comparaison longueur par label
            fake_lengths = df_clean[df_clean[label_column] == 0]['text_length']
            real_lengths = df_clean[df_clean[label_column] == 1]['text_length']
            
            axes[1, 1].hist(fake_lengths, bins=30, alpha=0.7, label='Fake News', color='lightcoral')
            axes[1, 1].hist(real_lengths, bins=30, alpha=0.7, label='Real News', color='lightgreen')
            axes[1, 1].set_title('Longueur des Textes par Label')
            axes[1, 1].set_xlabel('Longueur (caractères)')
            axes[1, 1].set_ylabel('Fréquence')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Sauvegarder
            plot_path = self.plots_dir / 'data_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Afficher les statistiques
            self._log_data_statistics(df_clean, label_counts)
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des données: {str(e)}")
            return None
    
    def _log_data_statistics(self, df_clean: pd.DataFrame, label_counts: dict):
        """
        Log les statistiques des données
        """
        logger.info(f"\n{'='*50}")
        logger.info("STATISTIQUES DES DONNÉES")
        logger.info(f"{'='*50}")
        logger.info(f"Nombre total d'échantillons: {len(df_clean)}")
        logger.info(f"Fake News: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df_clean)*100:.1f}%)")
        logger.info(f"Real News: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df_clean)*100:.1f}%)")
        logger.info(f"Longueur moyenne des textes: {df_clean['text_length'].mean():.0f} caractères")
        logger.info(f"Nombre moyen de mots: {df_clean['word_count'].mean():.0f} mots")
        logger.info(f"Longueur médiane des textes: {df_clean['text_length'].median():.0f} caractères")
    
    def create_performance_report(self, 
                                 df_clean: pd.DataFrame,
                                 text_column: str,
                                 label_column: str) -> str:
        """
        Crée un rapport détaillé des performances
        
        Args:
            results_df: DataFrame avec les résultats
            df_clean: DataFrame nettoyé
            text_column: Nom de la colonne de texte
            label_column: Nom de la colonne de labels
            
        Returns:
            str: Chemin vers le rapport
        """
        try:
            report_dir = Path("reports")
            report_dir.mkdir(exist_ok=True)
            
            # Créer le rapport
            report_path = report_dir / "performance_report.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Rapport de Performance - Détection de Fake News\n\n")
                f.write(f"**Date de génération:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Informations sur les données
                f.write("## 📊 Informations sur les Données\n\n")
                f.write(f"- **Nombre total d'échantillons:** {len(df_clean)}\n")
                label_dist = df_clean[label_column].value_counts()
                f.write(f"- **Fake News:** {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(df_clean)*100:.1f}%)\n")
                f.write(f"- **Real News:** {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(df_clean)*100:.1f}%)\n")
                f.write(f"- **Longueur moyenne des textes:** {df_clean[text_column].str.len().mean():.0f} caractères\n")
                f.write(f"- **Nombre moyen de mots:** {df_clean[text_column].str.split().str.len().mean():.0f} mots\n\n")
                
                # Recommandations
                f.write("## 💡 Recommandations\n\n")
                
                # Analyse de l'équilibrage
                balance_ratio = label_dist.get(1, 0) / label_dist.get(0, 1) if label_dist.get(0, 0) > 0 else 0
                if 0.8 <= balance_ratio <= 1.2:
                    f.write("- ✅ **Données équilibrées** - Pas besoin de techniques de rééquilibrage\n")
                else:
                    f.write("- ⚠️ **Données déséquilibrées** - Considérer SMOTE ou class_weight\n")
                
                # Analyse des performances
            """
                 avg_f1 = results_df['f1_score'].mean()
                if avg_f1 > 0.85:
                    f.write("- ✅ **Excellentes performances** - Modèles prêts pour la production\n")
                elif avg_f1 > 0.75:
                    f.write("- ⚠️ **Bonnes performances** - Optimisation possible avec feature engineering\n")
                else:
                    f.write("- ❌ **Performances à améliorer** - Revoir le preprocessing et les features\n")
               
                f.write("\n")
                f.write("## 📈 Visualisations\n\n")
                f.write("- Graphique comparatif des modèles\n")
             """
            logger.info(f"Rapport de performance créé: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du rapport: {str(e)}")
            return None
    
    def log_artifacts_to_mlflow(self, 
                               data_analysis_path: Optional[str] = None):
        """
        Enregistre les artefacts de visualisation dans MLflow
        
        Args:
            data_analysis_path: Chemin vers l'analyse des données
            learning_curve_path: Chemin vers les courbes d'apprentissage
            confusion_matrix_path: Chemin vers la matrice de confusion
            roc_curve_path: Chemin vers la courbe ROC
            comparison_plot_path: Chemin vers le graphique comparatif
        """
        try:
            if data_analysis_path:
                mlflow.log_artifact(data_analysis_path, "data_analysis")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement des artefacts: {str(e)}")

def load_data(file_path: str):
    """Fonction pour charger les données"""
    processor = DataVisualizer()
    return processor.load_data(file_path)


if __name__=="__main__":
    visualizer = DataVisualizer()

    try:                 
        # load data 
        df = visualizer.load_data("/home/dona-erick/Fake_News/Data_cleaned/data_cleaned.csv")
        # analyse et visualisation
        analysis = visualizer.analyze_and_visualize_data(df=df)
        logger.info(f"Analyse des données :{analysis}")
        # les logs 
        #label_counts = visualizer.analyze_and_visualize_data(df)
        #logs = visualizer._log_data_statistics(df, label_counts=label_counts)
        #logger.info("Logs des données :", logs)
        
        report = visualizer.create_performance_report(
                df_clean=df,
                text_column="cleaned_text",
                label_column="label"
            )
        print("Rapport :", report)

        visualizer.log_artifacts_to_mlflow(analysis)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution : {e}")
