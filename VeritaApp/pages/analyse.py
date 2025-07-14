import streamlit as st
import time, requests
from pages.home import home_page
def analyse_page():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700&display=swap');

            /* Styles généraux pour le corps et les éléments Streamlit */
            html, body {
                /* Dégradé de gris élégant */
                background: linear-gradient(135deg, #e0e0e0 0%, #f0f0f0 100%);
                font-family: 'Montserrat', serif; /* Police moderne et lisible */
                color: #333333; /* Texte gris foncé pour la lisibilité */
                line-height: 1.6;
            }
            .main {
                background: none; /* Permet au dégradé du body de transparaître */
                padding: 20px; /* Ajoute un peu de padding au contenu principal */
            }
            .css-h5rpku { /* Sélecteur spécifique pour le conteneur principal du contenu Streamlit */
                background-color: rgba(255, 255, 255, 0.9); /* Fond blanc semi-transparent pour les sections de contenu */
                border-radius: 12px;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08); /* Ombre douce et professionnelle */
                padding: 30px;
                margin-top: 30px;
            }
                /* Titres de section pour le contenu */
            .section-title {
                font-family: 'Playfair Display', serif;
                color: #2c3e50; /* Bleu très foncé, presque noir */
                font-size: 2.5em;
                font-weight: 700;
                margin-top: 50px;
                margin-bottom: 25px;
                text-align: center;
                position: relative; /* Pour le pseudo-élément */
            }
            .section-title::after {
                content: '';
                display: block;
                width: 80px; /* Longueur de la ligne */
                height: 4px; /* Épaisseur de la ligne */
                background-color: #3f51b5; /* Couleur de la ligne */
                margin: 10px auto 0; /* Centre la ligne sous le titre */
                border-radius: 2px;
                animation: expandLine 1.5s ease-out forwards;
            }
                 /* Sous-titres */
            .subtitle {
                font-family: 'Montserrat', sans-serif;
                font-size: 1.2em;
                color: #555555;
                max-width: 700px;
                line-height: 1.7;
                margin-bottom: 15px;
                animation: fadeIn 2s ease-in forwards;
            }
            .subtitle-agent { /* Utilisé pour des messages ou des slogans clés */
                font-family: 'Montserrat', sans-serif;
                font-size: 1.3em;
                font-weight: 600;
                color: #4a6a7a; /* Un gris-bleu harmonieux */
                margin-bottom: 20px;
                animation: fadeIn 2.5s ease-in forwards;
            }
            .stButton > button {
                background-color: #50a8ff;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 0.5em 1em;
                transition: 0.3s ease-in-out;
            }
            .stButton > button:hover {
                background-color: #2e86c1;
                transform: scale(1.02);
            }
            p {
                font-family: 'Montserrat', sans-serif;
                font-size: 1.05em;
                line-height: 1.7;
                color: #444444;
                text-align: justify;
                margin-bottom: 15px;
            }
                /* Optimisation pour les zones de texte et autres entrées */
            .stTextInput>div>div>input, .stTextArea>div>div>textarea {
                border-radius: 8px;
                border: 1px solid #ccc;
                padding: 10px 15px;
                box-shadow: inset 0 1px 3px rgba(0,0,0,0.08);
                transition: border-color 0.3s ease, box-shadow 0.3s ease;
            }
            .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
                border-color: #3f51b5;
                box-shadow: 0 0 0 3px rgba(63, 81, 181, 0.2); /* Halo de focus élégant */
                outline: none;
            }
                 /* Styles pour les messages d'info/succès/erreur (améliorés) */
            .stAlert {
                border-radius: 10px;
                padding: 20px;
                font-family: 'Montserrat', sans-serif;
                font-size: 1.05em;
                margin-top: 20px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }
            .stAlert.info { border-left: 6px solid #2196f3; background-color: #e3f2fd; color: #1e88e5; } /* Bleu clair */
            .stAlert.success { border-left: 6px solid #4caf50; background-color: #e8f5e9; color: #388e3c; } /* Vert clair */
            .stAlert.warning { border-left: 6px solid #ff9800; background-color: #fff3e0; color: #fb8c00; } /* Orange clair */
            .stAlert.error { border-left: 6px solid #f44336; background-color: #ffebee; color: #d32f2f; } /* Rouge clair */
            }
                    /* Animations */
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            @keyframes fadeInScale {
                from { opacity: 0; transform: scale(0.95); }
                to { opacity: 1; transform: scale(1); }
            }
            @keyframes slideInFromTop {
                from { opacity: 0; transform: translateY(-30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes expandLine {
                from { width: 0; }
                to { width: 80px; }
            }

                        
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='section-title'>🔎 Détecteur des Vraies Informations</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Bienvenue sur la page de détection ! Ici, vous pourrez soumettre le texte que vous souhaitez analyser.</p>", unsafe_allow_html=True)

    texte = st.text_area(
        label="📝 Texte d'actualité",
        placeholder="Copiez-collez ici le contenu ou le titre d'une actualité...",
        height=300
    )

    if st.button("Analyser"):
        if texte.strip():
            with st.spinner("Analyse en cours... Veuillez patienter."):
                try:
                    response = requests.post(
                        url="http://127.0.0.1:8000/predict",
                        json={
                            'text':texte,
                            'model_name':"model_linearsvc"
                        }
                          )
                    if response.status_code == 200:
                        result = response.json()

                        st.success("Analyse terminée ! Voici le résultat :")
                        st.markdown("---")
                        st.subheader("✅ Résultat de VeritaAI")
                        st.write(f"**Prédiction :** {'🟥 Fake News' if result['prediction'] == 0 else '🟩 Real News'}")
                        st.write(f"**Confiance :** {result['confidence']*100:.2f}%")
                        st.write(f"**Longueur du texte :** {result['text_length']} caractères")
                        st.write(f"**Modèle utilisé :** {result['model_used']}")
                    else:
                        st.error(f"Erreur de l'API : {response.status_code} - {response.text}")
                    st.markdown("---")
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Back to home", key="back_to_home"):
        st.rerun()
        home_page()
        st.stop()
if __name__ == "__main__":
    analyse_page()
