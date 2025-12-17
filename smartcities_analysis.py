"""
Módulo de análisis de sentimientos para Smart Cities
"""
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import os
from datetime import datetime

# Configurar NLTK
nltk.data.path.append('./nltk_data')

try:
    nltk.download('vader_lexicon', quiet=True)
except:
    # Si falla la descarga, intentar usar localmente
    pass

# Inicializar analizador
analyzer = SentimentIntensityAnalyzer()

# Datos de ejemplo iniciales (15 comentarios originales + 40 nuevos)
COMENTARIOS_EJEMPLO = [
    # Comentarios originales (15)
    "El transporte público es muy lento en horas punta",
    "Los buses nuevos han mejorado bastante el servicio",
    "Siempre hay mucha demora y desorden en los paraderos",
    "Me gusta que ahora las unidades estén más limpias",
    "El pasaje es caro para la calidad del servicio",
    "Los conductores manejan de forma irresponsable",
    "El sistema de transporte debería ser más puntual",
    "Buen servicio en algunas rutas, pero malo en otras",
    "El transporte público ha mejorado en los últimos meses",
    "Falta seguridad en los paraderos durante la noche",
    "Los horarios no se respetan y eso genera molestias",
    "El servicio es aceptable, aunque puede mejorar",
    "Muy mala experiencia usando el transporte público",
    "El viaje fue cómodo y rápido",
    "La atención al usuario es deficiente",
]

def traducir_es_en(texto):
    """Traducir español a inglés"""
    try:
        if not texto or not isinstance(texto, str):
            return texto
        
        # Detectar si hay texto en español
        palabras_es = ['el', 'la', 'los', 'las', 'es', 'en', 'y', 'de', 'que', 'se']
        tiene_espanol = any(palabra in texto.lower() for palabra in palabras_es)
        
        if tiene_espanol:
            return GoogleTranslator(source='es', target='en').translate(texto)
        return texto
    except Exception as e:
        print(f"⚠️ Error en traducción: {e}")
        return texto

def limpiar_texto(texto):
    """Limpiar texto para análisis"""
    if not texto or not isinstance(texto, str):
        return ""
    
    texto = texto.lower()
    # Mantener letras, números básicos y espacios
    texto = re.sub(r'[^a-z0-9áéíóúñü\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def clasificar_sentimiento(compound):
    """Clasificar según puntuación VADER"""
    if compound >= 0.05:
        return "Positivo"
    elif compound <= -0.05:
        return "Negativo"
    else:
        return "Neutro"

def analizar_sentimiento(comentario):
    """Analizar sentimiento de un comentario nuevo"""
    try:
        # Validar entrada
        if not comentario or not isinstance(comentario, str):
            raise ValueError("Comentario inválido")
        
        # Traducir si es necesario
        comentario_trad = traducir_es_en(comentario)
        
        # Limpiar
        comentario_limpio = limpiar_texto(comentario_trad)
        
        # Analizar con VADER
        scores = analyzer.polarity_scores(comentario_limpio)
        
        # Clasificar
        sentimiento = clasificar_sentimiento(scores['compound'])
        
        return {
            'comentario_original': comentario,
            'comentario_traducido': comentario_trad,
            'neg': float(scores['neg']),
            'neu': float(scores['neu']),
            'pos': float(scores['pos']),
            'compound': float(scores['compound']),
            'sentimiento': sentimiento
        }
    
    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        # Retornar valores por defecto en caso de error
        return {
            'comentario_original': comentario,
            'comentario_traducido': comentario,
            'neg': 0.0,
            'neu': 1.0,
            'pos': 0.0,
            'compound': 0.0,
            'sentimiento': "Neutro"
        }

def cargar_dataset():
    """Cargar dataset desde CSV"""
    csv_path = 'data/comentarios.csv'
    
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, encoding='utf-8')
            # Asegurar tipos de datos
            for col in ['neg', 'neu', 'pos', 'compound']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        else:
            return pd.DataFrame(columns=[
                'fecha', 'nombre', 'email', 'comentario', 
                'comentario_traducido', 'neg', 'neu', 'pos', 
                'compound', 'sentimiento'
            ])
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return pd.DataFrame()

def inicializar_dataset():
    """Inicializar dataset con ejemplos si está vacío"""
    csv_path = 'data/comentarios.csv'
    
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("📁 Inicializando dataset con ejemplos...")
        
        # Crear lista para nuevos datos
        nuevos_datos = []
        
        # Agregar comentarios de ejemplo
        for i, comentario in enumerate(COMENTARIOS_EJEMPLO[:15]):  # Solo primeros 15
            resultado = analizar_sentimiento(comentario)
            
            nuevo_registro = {
                'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'nombre': 'Ejemplo',
                'email': '',
                'comentario': comentario,
                'comentario_traducido': resultado['comentario_traducido'],
                'neg': resultado['neg'],
                'neu': resultado['neu'],
                'pos': resultado['pos'],
                'compound': resultado['compound'],
                'sentimiento': resultado['sentimiento']
            }
            nuevos_datos.append(nuevo_registro)
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(nuevos_datos)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Dataset inicializado con {len(df)} comentarios de ejemplo")

def agregar_comentario(comentario, nombre="Anónimo", email="", resultado_analisis=None):
    """Agregar nuevo comentario al dataset"""
    try:
        if resultado_analisis is None:
            resultado_analisis = analizar_sentimiento(comentario)
        
        # Crear nueva fila
        nueva_fila = {
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'nombre': nombre if nombre else 'Anónimo',
            'email': email,
            'comentario': comentario,
            'comentario_traducido': resultado_analisis['comentario_traducido'],
            'neg': resultado_analisis['neg'],
            'neu': resultado_analisis['neu'],
            'pos': resultado_analisis['pos'],
            'compound': resultado_analisis['compound'],
            'sentimiento': resultado_analisis['sentimiento']
        }
        
        # Cargar dataset existente
        df = cargar_dataset()
        
        # Agregar nueva fila
        df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
        
        # Guardar
        df.to_csv('data/comentarios.csv', index=False, encoding='utf-8')
        
        print(f"✅ Comentario agregado: {comentario[:50]}...")
        return True
    
    except Exception as e:
        print(f"❌ Error agregando comentario: {e}")
        return False

def obtener_estadisticas():
    """Obtener estadísticas del dataset"""
    df = cargar_dataset()
    
    if df.empty:
        return {
            'total': 0,
            'positivos': 0,
            'negativos': 0,
            'neutros': 0,
            'promedio_compound': 0.0
        }
    
    stats = df['sentimiento'].value_counts()
    
    return {
        'total': len(df),
        'positivos': int(stats.get('Positivo', 0)),
        'negativos': int(stats.get('Negativo', 0)),
        'neutros': int(stats.get('Neutro', 0)),
        'promedio_compound': float(df['compound'].mean() if 'compound' in df.columns else 0.0)
    }
