from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    print("🔵 RUTA / ACCEDIDA - Sirviendo index.html")
    try:
        return render_template('index.html')
    except Exception as e:
        return f"ERROR con template: {str(e)}"

@app.route('/test')
def test():
    return "🎉 ¡FLASK FUNCIONA! Ve a <a href='/'>Inicio</a>"

if __name__ == '__main__':
    print("=" * 60)
    print("🧪 TEST APP MÍNIMA")
    print("=" * 60)
    print(f"📂 Templates: {os.listdir('templates')}")
    print(f"🌐 URL: http://localhost:5000")
    print(f"🌐 Test: http://localhost:5000/test")
    print("=" * 60)
    print("Presiona CTRL+C para detener")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=True,
        use_reloader=False  # Desactiva reloader para ver mensajes
    )
