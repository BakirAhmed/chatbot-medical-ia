"""
CHATBOT MÉDICAL COMPLET
- Détection des symptômes avec sévérité
- Description de la maladie
- 3 précautions à prendre
- Interface bilingue (FR/EN)
"""

import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings("ignore")

# ============================================
# DICTIONNAIRES DE TRADUCTION
# ============================================

SYMPTOM_SYNONYMS = {
    'itching': ['démangeaison', 'gratte', 'prurit', 'ça gratte', 'itching'],
    'skin_rash': ['rougeur', 'éruption', 'bouton', 'plaque rouge', 'peau rouge', 'rash'],
    'nodal_skin_eruptions': ['nodule', 'éruption nodulaire', 'nodal'],
    'continuous_sneezing': ['éternuement', 'éternue', 'sternutation', 'sneeze', 'sneezing'],
    'shivering': ['frisson', 'tremble', 'tremblement', 'shiver'],
    'chills': ['froid', 'frisson de froid', 'chill'],
    'joint_pain': ['douleur articulaire', 'articulation douloureuse', 'genou douloureux', 'joint pain'],
    'stomach_pain': ['mal au ventre', 'douleur abdominale', 'ventre douloureux', 'stomach pain'],
    'acidity': ['brûlure d\'estomac', 'acidité', 'reflux', 'aigreur', 'acidity'],
    'ulcers_on_tongue': ['aphte', 'ulcère langue', 'plaie langue', 'ulcer tongue'],
    'muscle_wasting': ['fonte musculaire', 'perte de muscle', 'muscle wasting'],
    'vomiting': ['vomissement', 'vomi', 'vomiting', 'vomit'],
    'burning_micturition': ['brûlure en urinant', 'urine brûlante', 'burning urine'],
    'spotting_urination': ['uriner par gouttes', 'spotting urine'],
    'fatigue': ['fatigue', 'épuisement', 'lasitude', 'pas d\'énergie', 'tired', 'fatigue'],
    'weight_gain': ['prise de poids', 'grossir', 'weight gain'],
    'anxiety': ['anxiété', 'stress', 'angoissé', 'inquiet', 'anxiety'],
    'weight_loss': ['perte de poids', 'maigrir', 'weight loss'],
    'restlessness': ['agitation', 'nerveux', 'restless'],
    'lethargy': ['léthargie', 'apathie', 'somnolence', 'lethargy'],
    'cough': ['toux', 'tousser', 'cough'],
    'high_fever': ['forte fièvre', 'fièvre élevée', 'haute température', 'high fever'],
    'headache': ['mal de tête', 'céphalée', 'migraine', 'headache', 'head pain'],
    'nausea': ['nausée', 'mal au cœur', 'nausea'],
    'loss_of_appetite': ['perte d\'appétit', 'plus faim', 'loss appetite'],
    'back_pain': ['mal de dos', 'douleur lombaire', 'back pain'],
    'constipation': ['constipé', 'difficulté à aller à la selle', 'constipation'],
    'diarrhoea': ['diarrhée', 'selles liquides', 'diarrhoea', 'diarrhea'],
    'chest_pain': ['douleur thoracique', 'mal à la poitrine', 'chest pain'],
    'dizziness': ['vertige', 'étourdissement', 'tête qui tourne', 'dizzy', 'dizziness'],
    'depression': ['dépression', 'tristesse', 'depressed', 'depression'],
    'muscle_pain': ['courbature', 'douleur musculaire', 'muscle pain'],
    'polyuria': ['urine fréquente', 'pipi souvent', 'urination fréquente', 'polyuria'],
    'increased_appetite': ['beaucoup faim', 'faim excessive', 'augmentation appétit'],
}

SYMPTOM_FR = {
    'itching': 'Démangeaisons',
    'skin_rash': 'Rougeurs cutanées',
    'nodal_skin_eruptions': 'Éruptions nodulaires',
    'continuous_sneezing': 'Éternuements continus',
    'shivering': 'Frissons',
    'chills': 'Sensation de froid',
    'joint_pain': 'Douleurs articulaires',
    'stomach_pain': 'Douleurs abdominales',
    'acidity': 'Brûlures d\'estomac',
    'ulcers_on_tongue': 'Aphtes sur la langue',
    'muscle_wasting': 'Fonte musculaire',
    'vomiting': 'Vomissements',
    'burning_micturition': 'Brûlures en urinant',
    'fatigue': 'Fatigue',
    'weight_gain': 'Prise de poids',
    'anxiety': 'Anxiété',
    'weight_loss': 'Perte de poids',
    'cough': 'Toux',
    'high_fever': 'Forte fièvre',
    'headache': 'Maux de tête',
    'nausea': 'Nausées',
    'loss_of_appetite': 'Perte d\'appétit',
    'back_pain': 'Mal de dos',
    'constipation': 'Constipation',
    'diarrhoea': 'Diarrhée',
    'chest_pain': 'Douleur thoracique',
    'dizziness': 'Vertiges',
}

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================

def load_data():
    """Charge les datasets"""
    try:
        training = pd.read_csv('Data/Training.csv')
        testing = pd.read_csv('Data/Testing.csv')
        print("✅ Données chargées")
        print(f"   Training: {training.shape[0]} lignes")
        print(f"   Testing:  {testing.shape[0]} lignes")
        return training, testing
    except Exception as e:
        print(f"❌ Erreur chargement données: {e}")
        return None, None

def load_severity_dict():
    """Charge les sévérités des symptômes"""
    severity = {}
    try:
        with open('MasterData/Symptom_severity.csv', 'r') as f:
            for line in f:
                if ',' in line:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        symptom = parts[0].strip()
                        try:
                            severity[symptom] = int(parts[1])
                        except:
                            severity[symptom] = 5
        print(f"✅ Sévérités chargées: {len(severity)} symptômes")
    except Exception as e:
        print(f"⚠️ Fichier sévérité non trouvé: {e}")
        # Valeurs par défaut
        default = {'stomach_pain': 6, 'acidity': 5, 'ulcers_on_tongue': 6, 
                   'vomiting': 7, 'cough': 3, 'fatigue': 4, 'headache': 5}
        severity.update(default)
    return severity

def load_descriptions():
    """Charge les descriptions des maladies"""
    descriptions = {}
    try:
        with open('MasterData/symptom_Description.csv', 'r') as f:
            for line in f:
                if ',' in line:
                    parts = line.strip().split(',', 1)
                    if len(parts) >= 2:
                        descriptions[parts[0]] = parts[1]
        print(f"✅ Descriptions chargées: {len(descriptions)} maladies")
    except Exception as e:
        print(f"⚠️ Fichier descriptions non trouvé: {e}")
        # Descriptions par défaut
        descriptions['GERD'] = "Le reflux gastro-œsophagien est une remontée acide de l'estomac vers l'œsophage, causant brûlures et irritations."
        descriptions['Fungal infection'] = "Infection causée par des champignons microscopiques affectant généralement la peau."
        descriptions['Allergy'] = "Réaction excessive du système immunitaire à une substance normalement inoffensive."
        descriptions['Diabetes'] = "Maladie caractérisée par un excès de sucre dans le sang."
        descriptions['Hypertension'] = "Pression sanguine anormalement élevée dans les artères."
    return descriptions

def load_precautions():
    """Charge les précautions (3 par maladie)"""
    precautions = {}
    try:
        with open('MasterData/symptom_precaution.csv', 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if parts:
                    disease = parts[0]
                    precautions[disease] = parts[1:4] if len(parts) >= 4 else []
        print(f"✅ Précautions chargées: {len(precautions)} maladies")
    except Exception as e:
        print(f"⚠️ Fichier précautions non trouvé: {e}")
        # Précautions par défaut (3 par maladie)
        precautions['GERD'] = [
            "Mangez en petites portions",
            "Ne vous allongez pas après manger (attendez 2-3h)",
            "Évitez café, alcool, aliments gras et épicés"
        ]
        precautions['Fungal infection'] = [
            "Gardez la zone atteinte propre et sèche",
            "Utilisez une crème antifongique prescrite",
            "Portez des vêtements amples en coton"
        ]
        precautions['Allergy'] = [
            "Identifiez et évitez l'allergène",
            "Prenez un antihistaminique si prescrit",
            "Gardez votre environnement propre"
        ]
    return precautions

def train_model(training, testing):
    """Entraîne le modèle"""
    cols = training.columns[:-1]
    x_train = training[cols]
    y_train = training['prognosis']
    x_test = testing[cols]
    y_test = testing['prognosis']
    
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train_enc)
    
    score = model.score(x_test, y_test_enc)
    print(f"✅ Modèle entraîné (accuracy: {score:.2%})")
    
    return model, le, cols

# ============================================
# DÉTECTION DES SYMPTÔMES
# ============================================

def extract_symptoms(text, all_symptoms):
    """Extrait les symptômes du texte utilisateur"""
    text = text.lower()
    detected = []
    
    for symptom in all_symptoms:
        if symptom.replace('_', ' ') in text or symptom in text:
            detected.append(symptom)
            continue
        if symptom in SYMPTOM_SYNONYMS:
            for synonym in SYMPTOM_SYNONYMS[symptom]:
                if synonym.lower() in text:
                    detected.append(symptom)
                    break
    return list(set(detected))

def symptoms_to_vector(detected, all_symptoms):
    """Convertit en vecteur binaire"""
    return [1 if s in detected else 0 for s in all_symptoms]

def predict_disease(model, le, vector, all_symptoms):
    """Prédit la maladie"""
    df = pd.DataFrame([vector], columns=all_symptoms)
    pred_enc = model.predict(df)[0]
    return le.inverse_transform([pred_enc])[0]

# ============================================
# AFFICHAGE
# ============================================

def display_detected_with_severity(detected, severity_dict):
    """Affiche les symptômes détectés avec leur sévérité"""
    print("\n" + "=" * 55)
    print("🔍 SYMPTÔMES DÉTECTÉS AVEC LEUR SÉVÉRITÉ")
    print("=" * 55)
    print(f"{'N°':<4} {'Symptôme':<30} {'Sévérité':<10}")
    print("-" * 55)
    
    if not detected:
        print("   ❌ Aucun symptôme reconnu")
        return False
    
    total = 0
    for i, sym in enumerate(detected, 1):
        name_fr = SYMPTOM_FR.get(sym, sym.replace('_', ' ').title())
        sev = severity_dict.get(sym, 3)
        total += sev
        bar = "█" * min(sev, 10) + "░" * (10 - min(sev, 10))
        print(f"{i:<4} {name_fr:<30} {sev:<3}/10    {bar}")
    
    avg = total / len(detected)
    print("-" * 55)
    print(f"📊 Sévérité moyenne: {avg:.1f}/10")
    
    if avg >= 7:
        print("⚠️ ALERTE: Sévérité élevée! Consultez rapidement.")
    elif avg >= 5:
        print("⚠️ Attention: Sévérité modérée. Surveillez vos symptômes.")
    else:
        print("✅ Sévérité faible.")
    
    return True

def display_disease_info(disease, descriptions, precautions):
    """Affiche la description et les précautions de la maladie"""
    print("\n" + "=" * 55)
    print("🏥 DIAGNOSTIC")
    print("=" * 55)
    
    # Nom de la maladie
    disease_fr = {
        'GERD': 'Reflux gastro-œsophagien',
        'Fungal infection': 'Infection fongique',
        'Allergy': 'Allergie',
        'Diabetes': 'Diabète',
        'Hypertension': 'Hypertension'
    }.get(disease, disease)
    
    print(f"\n🩺 Maladie suspectée : {disease_fr}")
    
    # Description
    print("\n📖 DESCRIPTION:")
    desc = descriptions.get(disease, "Consultez un médecin pour un diagnostic précis.")
    print(f"   {desc}")
    
    # 3 précautions
    prec_list = precautions.get(disease, [
        "Consultez un médecin",
        "Suivez le traitement prescrit",
        "Reposez-vous suffisamment"
    ])[:3]
    
    print("\n🛡️ PRÉCAUTIONS (3 recommandations):")
    for i, p in enumerate(prec_list, 1):
        if p and p.strip():
            print(f"   {i}. {p.strip()}")
    
    print("\n" + "=" * 55)
    print("⚠️  Ce diagnostic est indicatif. Consultez un médecin.")
    print("=" * 55)

# ============================================
# INTERFACE PRINCIPALE
# ============================================

def main():
    print("=" * 60)
    print("🏥 CHATBOT MÉDICAL - ASSISTANT DE DIAGNOSTIC")
    print("=" * 60)
    print("\n🤖 Décrivez vos symptômes, j'identifierai la maladie possible")
    print("   avec sévérité, description et précautions.\n")
    
    # Chargement
    training, testing = load_data()
    if training is None:
        return
    
    model, le, cols = train_model(training, testing)
    all_symptoms = list(cols)
    
    severity_dict = load_severity_dict()
    descriptions = load_descriptions()
    precautions = load_precautions()
    
    print(f"\n📋 Base: {len(all_symptoms)} symptômes reconnus\n")
    
    # Boucle principale
    while True:
        print("-" * 55)
        print("\n👉 Décrivez vos symptômes (ou 'quit' pour quitter)")
        print("   Ex: 'J'ai mal au ventre, des brûlures, des aphtes, je vomis et je tousse'")
        
        user_input = input("\n> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Prenez soin de vous !")
            break
        
        if not user_input:
            print("❌ Veuillez décrire vos symptômes")
            continue
        
        # Extraction et affichage
        detected = extract_symptoms(user_input, all_symptoms)
        
        if not display_detected_with_severity(detected, severity_dict):
            continue
        
        # Prédiction
        vector = symptoms_to_vector(detected, all_symptoms)
        disease = predict_disease(model, le, vector, all_symptoms)
        
        # Affichage description + précautions
        display_disease_info(disease, descriptions, precautions)
        
        print("\n💡 Tapez 'quit' pour quitter ou continuez.\n")

if __name__ == "__main__":
    main()