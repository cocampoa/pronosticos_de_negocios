# pronosticos_de_negocios
Guía de referencia rápida para R: guía de referencia rápida (Cheat Sheet) muy completa para R, enfocada en análisis de datos, econometría y series de tiempo.
# R Quick Reference Guide: De Estadística Básica a Series de Tiempo

![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Esta es una guía de referencia rápida ("Cheat Sheet") desarrollada en R que abarca desde los fundamentos de la manipulación de datos y estadística descriptiva hasta modelos avanzados de regresión múltiple y análisis de series de tiempo.

## 🚀 Contenido de la Guía

El script principal está organizado de forma lógica para facilitar la consulta rápida de sintaxis y fórmulas:

1.  **Configuración:** Manejo de directorios y paquetes esenciales.
2.  **Manejo de Datos:** Lectura/escritura y conversión de tipos (factores, fechas, data frames).
3.  **Álgebra Lineal:** Operaciones con matrices y vectores.
4.  **Estadística Descriptiva:** Medidas de tendencia central, dispersión y correlación.
5.  **Probabilidad:** Distribuciones Normal y t-Student.
6.  **Regresión Lineal:** Modelos simples, múltiples, interacción y diagnóstico de supuestos.
7.  **Series de Tiempo:** Métodos Naive, descomposición manual/automática y pronósticos.
8.  **Inferencia:** Intervalos de confianza, predicción y pruebas de hipótesis (t-test, F-test, Jarque-Bera).

## 🛠️ Requisitos

Para ejecutar todo el código, asegúrate de tener instalados los siguientes paquetes en R:

```r
install.packages(c("tidyverse", "forecast", "tseries", "fpp2", "MASS", 
                   "lmtest", "moments", "lm.beta", "TSA"))
