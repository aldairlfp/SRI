<h1><center>Informe de Sistemas de Recuperación de Información</center></h1>
<h1><center>Universidad de la Habana</center></h1>

<center>
<h2>
Integrantes
</h2>
<h3>  
Aldair Alfonso Pérez y Mauro Bolado Vizoso
</h3>
</center>


<div style="page-break-after: always;"></div>

## Concepto

La recuperación de información es el conjunto de actividades orientadas a facilitar la localización de determinados datos u objetos, y las interrelaciones que estos tienen a su vez con otros. Existen varias disciplinas vinculadas a esta actividad como la lingüística, la documentación o la informática.

Aunque tradicionalmente se limitaba a la recuperación de documentos escritos, el término se redefinió para incorporar la creciente aparición de materiales multimedia. Asi, los nuevos buscadores de información en Internet, que originariamente buscaban textos, expandieron su actividad a imágenes, videos o audios. De esta forma términos como Recuperación de textos, recuperación documental y recuperación de información son utilizados como equivalentes.

Por otro lado, la necesidad de localizar datos concretos ha ido expandiendo su área de actuación. En la actualidad se está migrando desde la recuperación de documentos a la recuperación pregunta-respuesta, que responden con el dato concreto y no con el conjunto de documentos que posiblemente contenga este dato.

## Preprocesamiento de los datos

Como las collecciones de datos pueden ser diferentes, hay que transformar estas collecciones en algo que nuestro SRI pueda analizar. Por ahora solo se pueden transformar la colección *Cran*.

### Document

Primero que todo tenemos una clase `Document` que almacena el id, título, corpus, lenguaje, titulo normalizado y corpus normalizado, esto es para tener una mejor organización. Cuando se inicializa `Document` uno de los parametros es el procesador de texto que se utilizara en el documento.

### Collection

También tenemos una clase `Collection` que convierte las colecciones generales en listas de `Documento` con el método `parse`. También tenemos que `Cran` parse las collecciones de tipo *Cran*

#### Processor

Se utiliza el módulo `nltk` para tokenizar el corpus y el título y stemmizar las palabras.

## Modelo Vectorial:

### Concepto general:

La idea básica de este modelo de recuperación vectorial reside en la creación de la clase `VectorSpace` que implementa todo lo necesario para el modelo como: *term frequency*, *document frecuency*, *inverse document frecuency* y *term frequency inverse document frecuency*.

```python
class VectorSpace(object):

    def __init__(self, docs):
        M = len(docs)  # number of files in dataset
        self.docs = docs
        self._tf_dict = self._termFrequencyInDoc()  # returns term frequency
        self._df_dict = self._wordDocFre()  # returns document frequencies
        self._idf_dict = self._inverseDocFre(M)  # returns idf scores
        self._tf_idf = self._tfidf(docs)  # returns tf-idf scores
        self.a = 0.5
```

### *Term frequency*

Calcula la frecuencia de las palabras en el documento y  lo guarda en una lista de diccionarios donde la *i-ésima* posición es el documento y el conteo de palabras ocurre en el diccionario.

```python
def _termFrequencyInDoc(self):  
    tf_docs = [{} for doc in self.docs]  
  
    for doc in self.docs:  
        for i, word in enumerate(doc.norm_corpus):  
            if word in tf_docs[i]:  
                tf_docs[i][word] += 1  
            else:  
                tf_docs[i][word] = 1  
  
    return tf_docs
```

### *Document frequency*

Calcula la frecuencia de los documentos, es decir en cuantos documentos hace ocurrencia la palabra.

```python
def _wordDocFre(self):  
    df = {}  
    for doc in self.docs:  
        for word in doc.norm_corpus:  
            if word in df:  
                df[word] += 1  
            else:  
                df[word] = 1  
    return df
```

### *Inverse document frequency*

Sea *N* la cantidad total de documentos y $n_i$ la cantidad de documentos en los que aparece el término $t_i$. La frecuencia de ocurrencia de un término $t_i$ dentro de todos los documentos de la colección $idf_i$ esta dada por:

$idf_i=log\frac{N}{n_i}$

```python
def _inverseDocFre(self, length):  
    idf = {}  
    for word in self._df_dict:  
        idf[word] = np.log10(length / self._df_dict[word])  
    return idf
```

### Term frequency inverse document frequency

El peso del término $t_i$ en el documento $d_j$ está dado por:
$w_{i,j}=tf_{i,j} * idf_i$

```python
def _tfidf(self, doc):  
    tf_idf_scr = [{} for doc in self.docs]  
    for i, doc in enumerate(self.docs):  
        for word in doc.norm_corpus:  
            try:  
                tf = self._tf_dict[i][word]  
                idf = self._idf_dict[word]  
                tf_idf_scr[i][word] = tf * idf  
            except KeyError:  
                pass  
    return tf_idf_scr
```

### Ranking

Luego de tener calculado el elemento anterior se procede a calcular los pesos de la consulta con la constante de suavizado `a` y para calcular la similitud entre la consulta y los documentos se emplea el coseno del ángulo comprendido entre ambos vectores. Esto permite tener una ordenación por relevancia de los documentos.

## Uso

Para el uso del SRI es necesario tener el conjunto de palabras necesarias del módulo `nltk` para el tokenizer y el stemmer. Como solamente en esta versión del proyecto solo lee la colección Cran se debe copiar la colección en la carpeta raíz. Cuando todo este listo, ejecutar

```
python main.py
```