# pronosticos_de_negocios  
### Guía de referencia rápida en R para estadística, econometría y series de tiempo

## 📌 Contexto del proyecto

Este repositorio contiene una **Guía de Referencia Rápida (Cheat Sheet)** desarrollada en **R**, orientada al análisis de datos, la estadística inferencial, la econometría y el análisis de series de tiempo con enfoque en pronósticos de negocios.

El objetivo principal del proyecto es servir como una **herramienta de consulta técnica y conceptual**, no como un script ejecutable de principio a fin. Está diseñada para acompañar procesos de estudio, resolución de ejercicios, análisis reales y repaso metodológico, integrando tanto el **uso profesional de funciones automáticas de R** como la **comprensión matemática de los modelos subyacentes**.

La guía está pensada para:
- Estudiantes de estadística, economía, ingeniería o ciencia de datos.
- Analistas que trabajan con modelos inferenciales y series de tiempo.
- Profesionales que desean entender qué ocurre “detrás del código” sin perder eficiencia operativa.

---

## 🧠 Análisis y enfoque metodológico

La estructura del script responde a una progresión lógica del análisis estadístico clásico:

1. **Fundamentos operativos**
   - Configuración del entorno y gestión de paquetes.
   - Manejo de datos, tipos, fechas y estructuras base.

2. **Base matemática y estadística**
   - Álgebra lineal aplicada a modelos estadísticos.
   - Estadística descriptiva, correlación y covarianza.
   - Distribuciones de probabilidad (Normal y t-Student).

3. **Inferencia estadística**
   - Pruebas de hipótesis (t, F, correlación).
   - Intervalos de confianza y predicción.
   - Cálculo manual de estadísticos para reforzar la intuición teórica.

4. **Regresión lineal**
   - Regresión simple y múltiple.
   - Diagnóstico de supuestos (normalidad, homocedasticidad, independencia).
   - Interpretación de coeficientes, pruebas individuales y globales.
   - Construcción matricial del estimador MCO:  
     \[
     \hat{\beta} = (X'X)^{-1}X'y
     \]

5. **Variables categóricas e interacciones**
   - Manejo de factores y variables dummy.
   - Modelos con interacción y comparación de coeficientes.
   - Estandarización de betas para análisis de importancia relativa.

6. **Series de tiempo y pronósticos**
   - Creación y exploración de objetos `ts`.
   - Métodos Naive y Naive estacional.
   - Descomposición automática y manual (Tendencia, Estacionalidad, Ciclo).
   - Medidas de error (MAE, MSE, MAPE).
   - Pronósticos con intervalos de confianza.

Un principio central del proyecto es **combinar automatización y comprensión**:  
las funciones nativas de R (`lm()`, `predict()`, `confint()`, `accuracy()`) se utilizan junto a fórmulas manuales comentadas, permitiendo entender la mecánica estadística sin sacrificar productividad profesional.

---

## 📊 Contenido principal

La guía cubre, entre otros temas:

- Manejo y limpieza de datos
- Álgebra matricial aplicada a regresión
- Estadística descriptiva e inferencial
- Regresión lineal simple y múltiple
- Diagnóstico de modelos
- Variables categóricas e interacciones
- Series de tiempo y descomposición
- Pronósticos y evaluación de precisión

---

## 🛠️ Requisitos

Para ejecutar la mayoría de los ejemplos, se recomienda contar con los siguientes paquetes:

```r
install.packages(c(
  "tidyverse", "forecast", "tseries", "fpp2", "MASS",
  "lmtest", "moments", "lm.beta", "TSA"
))
