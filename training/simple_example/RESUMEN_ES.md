# Resumen del Entrenamiento del Modelo TTS (Español)

## Descripción General

Este directorio contiene un ejemplo completo de entrenamiento de un modelo TTS (Text-to-Speech) para Kani TTS, con el modelo entrenado guardado en el repositorio.

## ¿Qué se ha hecho?

### 1. Script de Entrenamiento (`train_model.py`)

Se ha creado un script completo de entrenamiento que:
- Crea un modelo TTS desde cero (3.38 millones de parámetros)
- Entrena el modelo con datos sintéticos para demostración
- Guarda el modelo entrenado en el repositorio

**Características del modelo:**
- Arquitectura: GPT-2 adaptada para TTS
- Capas: 4
- Dimensión de embedding: 256
- Cabezales de atención: 4
- Tamaño de vocabulario: 362 tokens

### 2. Modelo Entrenado Guardado

El modelo entrenado se encuentra en: `models/kani-tts-trained/`

**Archivos incluidos:**
- `model.safetensors` - Pesos del modelo (13 MB)
- `config.json` - Configuración del modelo
- `tokenizer_config.json` - Configuración del tokenizador
- `vocab.json` - Vocabulario
- `training_metadata.json` - Metadatos del entrenamiento

**Total del modelo:** ~52 MB (incluye checkpoint)

### 3. Dataset Utilizado

Para esta demostración se utilizó un **dataset sintético** con 10 muestras de texto:

1. "Hello world, this is a test."
2. "The quick brown fox jumps over the lazy dog."
3. "Machine learning is fascinating."
4. "Text to speech synthesis."
5. "Welcome to the future of AI."
6. "Natural language processing."
7. "Deep learning models are powerful."
8. "Speech recognition technology."
9. "Audio generation systems."
10. "Transformer architectures work well."

### 4. Información sobre Datasets Reales

Se ha incluido documentación completa sobre datasets públicos recomendados:

#### Datasets Recomendados:

1. **LJSpeech** - Dataset en inglés, un solo hablante
   - Duración: 24 horas
   - Muestras: 13,100 clips de audio
   - URL: https://keithito.com/LJ-Speech-Dataset/

2. **Common Voice** - Dataset multilingüe colaborativo
   - 100+ idiomas
   - Duración variable por idioma
   - URL: https://commonvoice.mozilla.org/

3. **LibriTTS** - Corpus en inglés con múltiples hablantes
   - Duración: 585 horas
   - Hablantes: 2,456
   - URL: https://www.openslr.org/60/

4. **VCTK** - Corpus en inglés con varios acentos
   - Duración: 44 horas
   - Hablantes: 110
   - URL: https://datashare.ed.ac.uk/handle/10283/3443

5. **M-AILABS** - Dataset multilingüe de audiolibros
   - Idiomas: 8 (inglés, alemán, español, francés, etc.)
   - Duración: 1000+ horas
   - URL: https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/

## Resultados del Entrenamiento

### Progreso del entrenamiento:
```
Época 0.2: pérdida=5.7586
Época 0.4: pérdida=5.6083
Época 0.6: pérdida=5.3949
Época 0.8: pérdida=5.3365
Época 1.0: pérdida=5.3464
```

**Pérdida final:** 5.551

### Pruebas del Modelo:
✓ El modelo se carga correctamente
✓ El tokenizador funciona
✓ La inferencia genera salida
✓ Tamaño del modelo: 3.38M parámetros

## Cómo Usar

### 1. Entrenar el modelo de demostración:

```bash
cd training/simple_example
pip install -r requirements.txt
python train_model.py
```

### 2. Probar el modelo entrenado:

```bash
python test_model.py
```

### 3. Entrenar con datos personalizados:

Para entrenar con tus propios datos de audio:

1. Prepara tu dataset:
```
data/
├── audio/
│   ├── muestra_001.wav
│   ├── muestra_002.wav
│   └── ...
└── metadata.csv
```

2. Formato del archivo metadata.csv:
```csv
audio_path,transcript
audio/muestra_001.wav,Hola mundo
audio/muestra_002.wav,Esta es una prueba
```

3. Usa el script de entrenamiento personalizado:
```bash
python train_with_custom_data.py \
  --audio_dir data/audio \
  --metadata data/metadata.csv \
  --output_dir models/mi-modelo \
  --epochs 5 \
  --batch_size 4
```

## Documentación Incluida

1. **README.md** - Guía de inicio rápido
2. **DATASET_INFO.md** - Información detallada sobre datasets
3. **TRAINING_SUMMARY.md** - Resumen técnico del entrenamiento
4. **RESUMEN_ES.md** - Este documento (resumen en español)

## Archivos Creados

Total de archivos: **33 archivos**

Estructura:
```
training/
├── README.md                      - Documentación principal
└── simple_example/
    ├── DATASET_INFO.md           - Información sobre datasets
    ├── README.md                 - Guía del ejemplo
    ├── TRAINING_SUMMARY.md       - Resumen del entrenamiento
    ├── RESUMEN_ES.md            - Este documento
    ├── requirements.txt          - Dependencias
    ├── train_model.py           - Script de entrenamiento
    ├── test_model.py            - Script de prueba
    ├── train_with_custom_data.py - Template para datos propios
    └── models/
        └── kani-tts-trained/     - Modelo entrenado (52 MB)
            ├── model.safetensors
            ├── config.json
            ├── tokenizer_config.json
            └── ...
```

## Notas Importantes

⚠️ **Importante:** Este es un modelo de demostración entrenado con datos sintéticos.

Para uso en producción:
1. Usa un dataset real de audio-texto
2. Entrena por más épocas (5-10+)
3. Usa un dataset más grande (2-10+ horas de audio)
4. Ajusta los hiperparámetros según tu caso de uso

## Próximos Pasos Recomendados

1. **Recopilar Datos Reales:**
   - Descarga LJSpeech para inglés
   - Usa Common Voice para otros idiomas
   - O crea tu propio dataset con Datamio

2. **Preparar el Dataset:**
   - Sigue la guía en DATASET_INFO.md
   - Formatea el audio a 22050 Hz WAV
   - Crea transcripciones precisas

3. **Entrenar Modelo de Producción:**
   - Usa los scripts como plantilla
   - Entrena por al menos 5-10 épocas
   - Monitorea la pérdida de validación
   - Usa GPU para entrenamiento más rápido

## Requisitos de Hardware

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| GPU        | 8GB VRAM | 16GB+ VRAM |
| RAM        | 16GB | 32GB+ |
| Almacenamiento | 10GB | 50GB+ |
| CPU        | 4 núcleos | 8+ núcleos |

## Recursos Adicionales

- [README Principal](../../README.md)
- [Ejemplo de Euskera](../../finetuning/euskera/)
- [Pipeline de Finetuning](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline)
- [Comunidad Discord](https://discord.gg/NzP3rjB4SB)

## Licencia

Apache 2.0 (igual que el repositorio principal de Kani TTS)

---

**Modelo Entrenado:** 7 de diciembre de 2025
**Plataforma:** GitHub Copilot
**Propósito:** Demostración y uso educativo
