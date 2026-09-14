# Guía de Laboratorio N.° 1 - Fundamentos y Entornos de Desarrollo en Python

Proyecto para la asignatura de Tecnologías Emergentes (ISO46B) - UNCP.

## Requisitos
* Python 3.10+
* Git

## Instalación
1. Crear y activar el entorno virtual:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
## Flujo de datos — Semana 2

Esta sección documenta el pipeline de datos construido en la Semana 2 (Librerías para Datos y Automatización).

**Fuente:** API pública Open-Meteo (`https://api.open-meteo.com/v1/forecast`), sin necesidad de clave de acceso. Se consulta el pronóstico de 7 días para Huancayo (latitud -12.07, longitud -75.21): temperatura máxima, temperatura mínima y precipitación diaria.

**Transformación:**
1. `clima.py` consume la API con `requests` (timeout de 5s y manejo de excepciones) y guarda la respuesta cruda en `pronostico_huancayo.json`.
2. La misma respuesta se convierte a `pronostico_huancayo.csv` con el módulo estándar `csv`.
3. `analisis.py` carga el CSV en un DataFrame de Pandas, agrega las columnas derivadas `amplitud_termica`, `dia_lluvioso` y `categoria` (frío/templado/cálido), y calcula un resumen agrupado por categoría con `groupby`.

**Salida:**
- `pronostico_huancayo.json` — respuesta cruda de la API (trazabilidad del dato original).
- `pronostico_huancayo.csv` — datos tabulares sin procesar.
- `pronostico_huancayo_procesado.csv` — datos con las columnas derivadas.
- `resumen_por_categoria.csv` — agregación por categoría de temperatura.

**Cómo reproducirlo:**
```bash
python clima.py
python analisis.py
```