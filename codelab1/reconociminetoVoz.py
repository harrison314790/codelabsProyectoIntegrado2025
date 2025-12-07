import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import tempfile, os

SRATE = 16000     # tasa de muestreo
DUR = 5

print("Grabando... habla ahora!")
audio = sd.rec(int(DUR*SRATE), samplerate=SRATE, channels=1, dtype='int16')
sd.wait()
print("Listo, procesando...")

# guarda a WAV temporal
tmp_wav = tempfile.mktemp(suffix=".wav")
write(tmp_wav, SRATE, audio)

# reconoce con SpeechRecognition
r = sr.Recognizer()
with sr.AudioFile(tmp_wav) as source:
    data = r.record(source)

try:
    texto = r.recognize_google(data, language="es-ES")
    print("Dijiste:", texto)
except sr.UnknownValueError:
    print("No se entendió el audio.")
except sr.RequestError as e:
    print("Error:", e)
finally:
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)

#---------- COMANDOS ----------

cmd = texto.lower()

if "hola" in cmd:
    print("¡Hola, bienvenido al curso!")

elif "abrir google" in cmd:
    import webbrowser
    webbrowser.open("https://www.google.com")

elif "abrir steam" in cmd:
    import subprocess
    posibles_rutas = [
        r"C:\Program Files (x86)\Steam\steam.exe",
        r"C:\Program Files\Steam\steam.exe"
    ]
    ruta_steam = next((p for p in posibles_rutas if os.path.exists(p)), None)
    if ruta_steam:
        try:
            subprocess.run([ruta_steam], check=False)
        except Exception as e:
            print("No se pudo abrir Steam:", e)
    else:
        print("No encontré steam.exe. Verifica la ruta de instalación.")

elif "clima" in cmd:
    import webbrowser
    webbrowser.open("https://www.accuweather.com/es/co/tulua/106801/weather-forecast/106801")

elif "ua" in cmd:
    import webbrowser
    webbrowser.open("https://youtube.com/clip/UgkxRXSPStMIvfOFXaoXWRq1DrOLopJTrHv_?si=xpBuixohfGWrxyTl")

elif "hora" in cmd:
    from datetime import datetime
    print("Hora actual:", datetime.now().strftime("%H:%M"))
else:
    print("Comando no reconocido.")