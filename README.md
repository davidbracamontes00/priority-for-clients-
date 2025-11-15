# 📦 LogisticMate — Priority for Clients  
### Optimización logística con Machine Learning y segmentación inteligente

**LogisticMate** es un sistema creado para analizar operaciones logísticas, evaluar la puntualidad de entregas, clasificar modos de envío y segmentar clientes o productos mediante Machine Learning.  
Incluye modelos supervisados, clustering optimizado y herramientas de predicción listas para producción.

---

## 🧠 Tecnologías utilizadas
- Python 3.9+
- Pandas, NumPy  
- Scikit-Learn  
- XGBoost  
- Matplotlib / Seaborn  
- Joblib  
- OpenPyXL  

## 📁 Estructura del proyecto
### ✔️ 1. Preprocesamiento completo
- Codificación de variables categóricas  
- Limpieza y enriquecimiento del dataset  
- Escalado y normalización  

### ✔️ 2. Análisis Exploratorio (EDA)
- Distribuciones  
- Correlaciones  
- Relación entre modo de envío, peso, descuento, etc.

### ✔️ 3. Modelos Supervisados
Incluye:
- **KNN**
- **SVM**
- **Random Forest**
- **XGBoost**

Cada modelo mide:
- Accuracy  
- F1 Score  
- Matriz de confusión  
- Reporte de clasificación  

### ✔️ 4. Clustering Optimizado (K-Means + Silhouette)
- Selección automática del número óptimo de clusters  
- Pipeline con imputación + escalado  
- Exportación de clusters a Excel  
- Guardado del modelo para producción

### ✔️ 5. Predicción de cluster para nuevos registros
Función lista para integrar en dashboards o APIs.

---

## 🛠️ Ejecución

Ejecutar el script principal:

```bash
python logisticmate_clean.py
