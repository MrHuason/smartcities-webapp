"""
Smart Cities - Análisis de Sentimientos Web App
Aplicación completa con Flask + VADER + Dashboard admin
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Necesario para servidor web
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

# Configurar salida inmediata
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

# Descargar recursos NLTK si no existen
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("📥 Descargando recursos NLTK (VADER lexicon)...")
    nltk.download('vader_lexicon', quiet=True)

# Configurar Flask
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'smartcities_2024_secret_key_dev'
app.config['UPLOAD_FOLDER'] = 'data/'

# Contraseña admin (en producción usar variables de entorno)
ADMIN_PASSWORD = "admin123"

# Asegurar carpetas necesarias
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

# Inicializar analizador VADER
analyzer = SentimentIntensityAnalyzer()

print("=" * 60)
print("🚀 SMART CITIES WEB APP - INICIANDO")
print("=" * 60)
print(f"📂 Directorio: {os.getcwd()}")
print(f"🐍 Python: {sys.version}")
print(f"📁 Templates: {os.listdir('templates') if os.path.exists('templates') else 'NO EXISTE'}")
print("=" * 60)

# ============================================================================
# FUNCIONES DE ANÁLISIS Y DATOS
# ============================================================================

def traducir_es_en(texto):
    """Traducir español a inglés para mejor análisis con VADER"""
    try:
        if not texto or not isinstance(texto, str):
            return texto
        
        # Detección simple de español
        palabras_es = ['el', 'la', 'los', 'las', 'es', 'en', 'y', 'de', 'que', 'se']
        texto_lower = texto.lower()
        
        # Si tiene palabras comunes en español, traducir
        tiene_espanol = any(palabra in texto_lower for palabra in palabras_es)
        if tiene_espanol and len(texto.split()) > 1:
            return GoogleTranslator(source='es', target='en').translate(texto)
        return texto
    except Exception as e:
        print(f"⚠️ Error en traducción: {e}")
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
    """Analizar sentimiento de un comentario usando VADER"""
    try:
        if not comentario or not isinstance(comentario, str):
            return {
                'comentario_original': comentario or '',
                'comentario_traducido': comentario or '',
                'neg': 0.0, 'neu': 1.0, 'pos': 0.0,
                'compound': 0.0, 'sentimiento': "Neutro"
            }
        
        # Traducir si es necesario
        comentario_trad = traducir_es_en(comentario)
        
        # Analizar con VADER
        scores = analyzer.polarity_scores(comentario_trad)
        
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
        return {
            'comentario_original': comentario,
            'comentario_traducido': comentario,
            'neg': 0.0, 'neu': 1.0, 'pos': 0.0,
            'compound': 0.0, 'sentimiento': "Neutro"
        }

def cargar_dataset():
    """Cargar dataset desde CSV"""
    csv_path = 'data/comentarios.csv'
    
    try:
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            df = pd.read_csv(csv_path, encoding='utf-8')
            return df
        else:
            # Dataset vacío con columnas correctas
            return pd.DataFrame(columns=[
                'id', 'fecha', 'nombre', 'email', 'comentario',
                'sentimiento', 'compound', 'positivo', 'negativo', 'neutro'
            ])
    except Exception as e:
        print(f"❌ Error cargando CSV: {e}")
        return pd.DataFrame()

def inicializar_dataset():
    """Inicializar dataset con comentarios de ejemplo"""
    csv_path = 'data/comentarios.csv'
    
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("📁 Inicializando dataset con ejemplos...")
        
        # Comentarios de ejemplo variados
        ejemplos = [
            ("El transporte público es muy lento en horas punta", "Carlos"),
            ("Los buses nuevos han mejorado bastante el servicio", "Ana"),
            ("Siempre hay mucha demora y desorden en los paraderos", "Luis"),
            ("Me gusta que ahora las unidades estén más limpias", "María"),
            ("El pasaje es caro para la calidad del servicio", "Pedro"),
            ("Excelente servicio en la nueva línea de metro", "Laura"),
            ("Los conductores son muy amables y profesionales", "Javier"),
            ("Falta más frecuencia en los horarios nocturnos", "Sofía"),
            ("La aplicación móvil funciona muy bien", "David"),
            ("Me siento seguro viajando en el transporte público", "Elena")
        ]
        
        datos = []
        for i, (comentario, nombre) in enumerate(ejemplos):
            resultado = analizar_sentimiento(comentario)
            datos.append({
                'id': i + 1,
                'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'nombre': nombre,
                'email': f'{nombre.lower()}@ejemplo.com',
                'comentario': comentario,
                'sentimiento': resultado['sentimiento'],
                'compound': resultado['compound'],
                'positivo': resultado['pos'],
                'negativo': resultado['neg'],
                'neutro': resultado['neu']
            })
        
        df = pd.DataFrame(datos)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Dataset inicializado con {len(df)} ejemplos")

def agregar_comentario(comentario, nombre="Anónimo", email=""):
    """Agregar nuevo comentario al dataset"""
    try:
        # Analizar sentimiento
        resultado = analizar_sentimiento(comentario)
        
        # Cargar dataset existente
        df = cargar_dataset()
        
        # Crear nuevo ID
        nuevo_id = df['id'].max() + 1 if not df.empty and 'id' in df.columns else 1
        
        # Nueva fila
        nueva_fila = {
            'id': nuevo_id,
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'nombre': nombre if nombre else 'Anónimo',
            'email': email,
            'comentario': comentario,
            'sentimiento': resultado['sentimiento'],
            'compound': resultado['compound'],
            'positivo': resultado['pos'],
            'negativo': resultado['neg'],
            'neutro': resultado['neu']
        }
        
        # Agregar y guardar
        df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
        df.to_csv('data/comentarios.csv', index=False, encoding='utf-8')
        
        print(f"✅ Comentario #{nuevo_id} agregado: {resultado['sentimiento']}")
        return True, resultado
    
    except Exception as e:
        print(f"❌ Error agregando comentario: {e}")
        return False, None

def obtener_estadisticas():
    """Obtener estadísticas para API"""
    df = cargar_dataset()
    
    if df.empty:
        return {
            'total': 0, 'positivos': 0, 'negativos': 0, 'neutros': 0,
            'promedio_compound': 0.0, 'status': 'empty'
        }
    
    stats = df['sentimiento'].value_counts()
    
    return {
        'total': len(df),
        'positivos': int(stats.get('Positivo', 0)),
        'negativos': int(stats.get('Negativo', 0)),
        'neutros': int(stats.get('Neutro', 0)),
        'promedio_compound': float(df['compound'].mean() if 'compound' in df.columns else 0.0),
        'status': 'ok'
    }

# ============================================================================
# RUTAS DE LA APLICACIÓN
# ============================================================================

@app.route('/')
def index():
    """Página principal - Formulario para usuarios"""
    stats = obtener_estadisticas()
    return render_template('index.html', 
                         title="Análisis de Transporte Público",
                         total=stats['total'],
                         positivos=stats['positivos'],
                         negativos=stats['negativos'],
                         neutros=stats['neutros'])

@app.route('/sobre')
def sobre():
    """Página sobre el proyecto"""
    return render_template('sobre.html', title="Sobre el Proyecto")

@app.route('/api/stats')
def get_stats():
    """API para estadísticas (AJAX)"""
    return jsonify(obtener_estadisticas())

@app.route('/submit', methods=['POST'])
def submit_comment():
    """Procesar nuevo comentario"""
    try:
        comentario = request.form.get('comentario', '').strip()
        nombre = request.form.get('nombre', 'Anónimo').strip()
        email = request.form.get('email', '').strip()
        
        # Validaciones
        if not comentario:
            return render_template('error.html', 
                                 message="Por favor, ingresa un comentario",
                                 error_code=400), 400
        
        if len(comentario) > 1000:
            return render_template('error.html',
                                 message="El comentario es demasiado largo (máximo 1000 caracteres)",
                                 error_code=400), 400
        
        # Agregar comentario
        success, resultado = agregar_comentario(comentario, nombre, email)
        
        if not success:
            return render_template('error.html',
                                 message="Error al guardar el comentario",
                                 error_code=500), 500
        
        # Mostrar resultado
        return render_template('result.html', 
                             comentario=comentario,
                             nombre=nombre,
                             sentimiento=resultado['sentimiento'],
                             score=resultado['compound'],
                             positivo=resultado['pos'],
                             negativo=resultado['neg'],
                             neutro=resultado['neu'])
    
    except Exception as e:
        return render_template('error.html', 
                             message=f"Error: {str(e)}",
                             error_code=500), 500

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Login para administrador"""
    if request.method == 'POST':
        password = request.form.get('password')
        if password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            session['admin_login_time'] = datetime.now().isoformat()
            return redirect(url_for('admin_dashboard'))
        return render_template('admin_login.html', 
                             error="Contraseña incorrecta"), 401
    
    return render_template('admin_login.html', error=None)

@app.route('/admin')
def admin_dashboard():
    """Dashboard del administrador"""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    try:
        # Cargar datos
        df = cargar_dataset()
        
        if df.empty:
            return render_template('admin.html',
                                 total_comentarios=0,
                                 stats={},
                                 ultimos_comentarios=[],
                                 df_preview=[],
                                 plot_url=None)
        
        # Estadísticas
        stats = df['sentimiento'].value_counts()
        
        # Crear gráficos
        plot_url = None
        if len(df) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Colores para sentimientos
            colores = {
                "Positivo": "#28a745",  # Verde
                "Negativo": "#dc3545",  # Rojo
                "Neutro": "#6c757d"     # Gris
            }
            
            # 1. Gráfico de barras
            bars = ax1.bar(stats.index, stats.values, 
                          color=[colores.get(s, "#007bff") for s in stats.index])
            ax1.set_title("Distribución de Sentimientos", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Sentimiento", fontsize=12)
            ax1.set_ylabel("Número de Comentarios", fontsize=12)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Agregar valores en barras
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=11)
            
            # 2. Gráfico de torta
            ax2.pie(stats.values, labels=stats.index, autopct='%1.1f%%',
                   colors=[colores.get(s, "#007bff") for s in stats.index],
                   startangle=90, explode=[0.05, 0.05, 0.05],
                   shadow=True, textprops={'fontsize': 11})
            ax2.set_title("Proporción de Sentimientos", fontsize=14, fontweight='bold')
            
            # Ajustar layout
            plt.tight_layout()
            
            # Convertir a base64 para HTML
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=80, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
        
        # Preparar datos para tabla
        df_sorted = df.sort_values('fecha', ascending=False)
        ultimos_comentarios = df_sorted.head(10).to_dict('records')
        df_preview = df_sorted.head(20).to_dict('records')
        
        return render_template('admin.html',
                             plot_url=plot_url,
                             total_comentarios=len(df),
                             stats=stats.to_dict(),
                             ultimos_comentarios=ultimos_comentarios,
                             df_preview=df_preview)
    
    except Exception as e:
        return render_template('error.html', 
                             message=f"Error en dashboard: {str(e)}",
                             error_code=500), 500

@app.route('/admin/export/<format>')
def export_data(format):
    """Exportar datos en diferentes formatos"""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    try:
        df = cargar_dataset()
        
        if format == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False, encoding='utf-8')
            output.seek(0)
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'comentarios_smartcities_{datetime.now().strftime("%Y%m%d")}.csv'
            )
        
        elif format == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Comentarios')
            output.seek(0)
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'comentarios_smartcities_{datetime.now().strftime("%Y%m%d")}.xlsx'
            )
        
        else:
            return render_template('error.html',
                                 message="Formato no soportado",
                                 error_code=400), 400
    
    except Exception as e:
        return render_template('error.html',
                             message=f"Error al exportar: {str(e)}",
                             error_code=500), 500

@app.route('/admin/delete/<int:id>', methods=['POST'])
def delete_comment(id):
    """Eliminar comentario (solo admin)"""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    try:
        df = cargar_dataset()
        if not df.empty and 'id' in df.columns:
            df = df[df['id'] != id]
            df.to_csv('data/comentarios.csv', index=False, encoding='utf-8')
        
        return redirect(url_for('admin_dashboard'))
    except Exception as e:
        return render_template('error.html',
                             message=f"Error al eliminar: {str(e)}",
                             error_code=500), 500

@app.route('/admin/logout')
def admin_logout():
    """Cerrar sesión admin"""
    session.pop('admin_logged_in', None)
    session.pop('admin_login_time', None)
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', 
                         message="Página no encontrada",
                         error_code=404), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html',
                         message="Error interno del servidor",
                         error_code=500), 500

# ============================================================================
# INICIO DE LA APLICACIÓN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌐 SERVIDOR INICIANDO...")
    print("   URL Principal: http://localhost:5000")
    print("   Login Admin:   http://localhost:5000/admin/login")
    print("   Password:      admin123")
    print("=" * 60)
    print("📊 Inicializando dataset...")
    
    # Inicializar dataset con ejemplos
    inicializar_dataset()
    
    print("✅ Dataset listo")
    print("🎯 Presiona CTRL+C para detener el servidor")
    print("=" * 60 + "\n")
    
    # Iniciar servidor Flask
    app.run(
        host='0.0.0.0',      # Accesible desde cualquier IP
        port=5000,           # Puerto
        debug=True,          # Modo debug (muestra errores)
        use_reloader=False,  # Sin recarga automática (para ver mensajes)
        threaded=True        # Múltiples solicitudes simultáneas
                                  )
