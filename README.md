# 🐾 Gatuno V15.1.1 Estable

Versión limpia y privada del pronosticador. Corrige el fallo de V15.0 que entrenaba con más de 12 mil partidos y sólo después comprobaba el calendario.

## Qué mejora V15.1.1

- Comprueba primero que existen partidos futuros.
- Divide la consulta de ESPN en bloques de siete días y recupera cada bloque por día si falla o llega vacío.
- Agrega Sofascore como tercer respaldo diario, sin usarlo para fabricar cuotas.
- Guarda la última agenda válida durante 72 horas para resistir caídas temporales.
- Conserva el histórico comprimido dentro de `embedded_history.py`; no necesita `historical_fallback.csv.gz`.
- Guarda el modelo calibrado y lo reutiliza mientras los datos no hayan cambiado.
- Cierra los resultados con una sola consulta por partido, no una por cada mercado.
- Mantiene el Vigilante Gatuno: propone ajustes temporales, pero nunca modifica criterios sin confirmación.

## Tiempo esperado

- Primera ejecución: puede tardar varios minutos porque construye variables y calibra los modelos.
- Ejecuciones posteriores: deben ser claramente más rápidas si la caché sigue vigente.
- Si no hay calendario, el proceso debe detenerse antes del entrenamiento y mostrar un mensaje para reintentar.

La duración depende de las fuentes públicas y del servidor gratuito. Una demora prolongada seguida de “No se obtuvo calendario” no es un resultado normal; V15.1 evita desperdiciar el entrenamiento en ese caso.

## Instalación limpia

1. Crea un repositorio nuevo en GitHub y elige **Private**.
2. Sube únicamente los archivos de esta carpeta; todos deben quedar en la raíz.
3. En Streamlit Community Cloud crea una aplicación nueva.
4. Selecciona el repositorio privado, la rama `main` y `app.py` como archivo principal.
5. Restringe el acceso de la aplicación a las personas autorizadas.
6. Pulsa **GENERAR PRONÓSTICOS V15.1.1** una sola vez y espera a que concluya.
7. No borres la versión anterior hasta comprobar que la nueva muestra partidos.

## Archivos

| Archivo | Función |
|---|---|
| `app.py` | Interfaz Gatuno. |
| `gatuno_forecaster.py` | Modelos, calibración y seis mercados. |
| `gatuno_data.py` | Datos, calendario, histórico y fuentes de respaldo. |
| `gatuno_context.py` | Contexto competitivo y calendario. |
| `gatuno_audit.py` | Congelación y cierre real de pronósticos. |
| `gatuno_adaptive.py` | Alertas y ajustes confirmables. |
| `gatuno_quality.py` | Vigilancia de calidad fuera de muestra. |
| `embedded_history.py` | Histórico autocontenido y verificado. |
| `requirements.txt` | Dependencias. |

## Uso responsable

El sistema es experimental. Los colores expresan evidencia del modelo, no garantizan aciertos ni rentabilidad. No persigue pérdidas y no debe usarse como sustituto de una decisión financiera responsable.
