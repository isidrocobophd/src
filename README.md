# Reto ML3: Clasificador de Estudiantes

## Descripción

Aplicación de Machine Learning para clasificar estudiantes que se graduaron o abandonaron en función de una lista de atributos. Realiza un análisis comparativo entre Logistic Regression y Decision Tree.

## Características

✅ **Preprocesamiento de datos**
- Reordena columnas (Target a primera posición)
- Elimina instancias con estado 'Enrolled'
- División estratificada: 80% entrenamiento, 10% validación, 10% prueba

✅ **Análisis de correlación**
- Calcula correlación de cada atributo con el objetivo
- Selecciona atributos con |correlación| ≥ 0.2
- Simplifica el modelo eliminando características débiles

✅ **Modelos entrenados**
- Logistic Regression con parámetros específicos
- Decision Tree Classifier con parámetros específicos

✅ **Evaluación comparativa**
- Precisión (Precision)
- Exactitud (Accuracy)
- Exhaustividad (Recall)

## Requisitos

- Python 3.8+
- pandas
- numpy
- scikit-learn

## Instalación

```bash
git clone https://github.com/isidrocobophd/src.git
cd src
git checkout ml-classifier
pip install -r requirements.txt
```

## Uso

1. Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/mahwiz/students-dropout-and-academic-success-dataset)
2. Coloca `data.csv` en el directorio raíz
3. Ejecuta: `python ml_classifier.py`

## Fases del Análisis

- **Fase 1**: Carga y Preprocesamiento
- **Fase 2**: División de Datos (80-10-10)
- **Fase 3 & 4**: Análisis de Correlación
- **Fase 5**: Entrenamiento de Modelos
- **Fase 6**: Evaluación y Comparación

## Autor

isidrocobophd
