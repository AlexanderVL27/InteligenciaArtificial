import string

# Función para limpiar el texto
def clean_text(text):
    # Definición de palabras de bajo valor (stop words) en español en la lista stop_words
    stop_words = set(["el", "la", "los", "las", "un", "una", "unos", "unas",  
                      "de", "del", "a", "ante", "bajo", "cabe", "con", "contra",
                      "desde", "en", "entre", "hacia", "hasta", "para", "por",
                      "según", "sin", "sobre", "tras", "y", "o", "pero", "ni", "más", "menos"])  

    # Creación de un traductor para eliminar la puntuación del texto
    translator = str.maketrans('', '', string.punctuation)
    cleaned_words = []

    # Separación del texto en oraciones usando el punto como delimitador
    sentences = text.split(".")
    for sentence in sentences:
        # Transformación de las oraciones a minúsculas, eliminación de la puntuación y separación en palabras individuales
        words = sentence.lower().translate(translator).split()
        # Filtrado de las palabras para eliminar aquellas que están en la lista de palabras de bajo valor
        cleaned_words.extend([word for word in words if word not in stop_words])

    return cleaned_words

# Función para contar la frecuencia de palabras
def count_words(word_list):
    word_counter = {}
    for word in word_list:
        # Conteo de la frecuencia de cada palabra en la lista de palabras
        if word in word_counter:
            word_counter[word] += 1
        else:
            word_counter[word] = 1
    return word_counter

# Función para calificar frases
def rate_sentences(text, word_counter):
    sentences = text.split(".")
    sentence_scores = {}

    for sentence in sentences:
        # Limpieza de las palabras en cada oración
        words = clean_text(sentence)
        # Cálculo de la puntuación de la oración sumando las frecuencias de las palabras presentes en el diccionario word_counter
        score = sum(word_counter[word] for word in words if word in word_counter)
        # Almacenamiento de la puntuación de cada oración en el diccionario sentence_scores
        sentence_scores[sentence] = score

    return sentence_scores

# Función para generar el resumen
def generate_summary(text, n):
    # Limpieza del texto original
    clean_words = clean_text(text)
    # Conteo de las palabras limpias
    word_counter = count_words(clean_words)
    # Calificación de las oraciones
    sentence_scores = rate_sentences(text, word_counter)

    # Clasificación de las oraciones según su puntuación en orden descendente
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    # Selección de las primeras n oraciones con mayor puntuación
    top_sentences = [sentence for sentence, score in sorted_sentences[:n]]

    # Creación del resumen uniendo las oraciones seleccionadas con un punto y aparte
    summary = '.\n\n'.join(top_sentences)
    return summary

# Texto de ejemplo
texto = """El mercado global está experimentando cambios significativos debido a la globalización y la digitalización. Las empresas se están adaptando a un entorno cada vez más interconectado y tecnológico. La transformación digital ha impulsado la automatización de procesos, el uso de análisis de datos y la aparición de nuevas formas de trabajo remoto. Además, la globalización ha permitido a las empresas expandirse más allá de las fronteras tradicionales, accediendo a nuevos mercados y oportunidades. Sin embargo, estos cambios también plantean desafíos, como la necesidad de mantener la ciberseguridad, la gestión de la información a gran escala y la preocupación por la brecha digital que deja rezagadas a ciertas poblaciones. La economía global se encuentra en un punto de inflexión, donde la adaptabilidad y la innovación son clave para la supervivencia y el crecimiento empresarial.
"""

# Número de frases para el resumen
numero_frases_resumen = 3

# Generación del resumen
resumen = generate_summary(texto, numero_frases_resumen)

# Guardar el resumen en un archivo
with open('resumen.txt', 'w') as file:
    file.write(resumen)

print("Resumen guardado en 'resumen.txt'")

