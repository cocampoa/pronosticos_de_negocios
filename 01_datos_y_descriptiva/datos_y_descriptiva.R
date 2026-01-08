# ============================================================
# MÓDULO 01 — DATOS Y ESTADÍSTICA DESCRIPTIVA
# ============================================================
# Este archivo forma parte del repositorio:
# pronosticos_de_negocios
#
# Propósito:
# - Guía de referencia conceptual y práctica
# - No está diseñado para ejecutarse de principio a fin
#
# Autor: Carlos Ocampo
# ============================================================

# --- 1. Paquetes y Configuración ----------------------------------------------
# Paquetes recomendados:
# boot, fma, forecast, greybox, ggplot2, lamW, lmtest, MASS, 
# naivebayes, olsrr, smooth, tseries, TSA, tidyverse, fpp2

# Gestión de Directorio:
# En lugar de rutas locales fijas, se recomienda usar Proyectos de RStudio (.Rproj)
getwd() 
# setwd("tu/ruta/aqui") # Comentado para portabilidad

# --- 2. Manejo de Datos (E/S) -------------------------------------------------

# Lectura de datos
datos_csv <- read.csv("rseIntro.csv")
datos_txt <- read.table("cableTV.dat", header = TRUE)

# Escritura de datos
write.csv(datos_csv, "cableTV_limpio.csv", row.names = FALSE)

# Conversión de tipos de datos
as.factor(dat)                          # Categorizar (Factores)
as.Date(dat$Col, format = "%d/%m/%Y")   # Formato fecha
as.character(dat$Col)                   # Texto
as.data.frame(matriz1)                  # De matriz a Data Frame

# --- 3. Matrices y Álgebra Lineal ---------------------------------------------

x1 <- c(7, 2, 9)                        # Vector columna
x2 <- c(1, 5, 8)
x3 <- t(x2)                             # Trasponer
suma <- x1 + x2                         # Suma/Resta (mismas dimensiones)
escalar <- x1 * 7                       # Vector por escalar
prod_punto <- x1 %*% x2                 # Producto matricial (escalar)
x1[2]                                   # Acceso al segundo elemento
combinado <- c(x1, x2)                  # Concatenación de vectores

# Creación de matrices
matriz1 <- matrix(c(1,1,1,2,2,2,4,3,5), nrow = 3, ncol = 3, byrow = TRUE)
multiplicacion <- matriz1 %*% matriz2   # Multiplicación matricial
matriz1[2, 3]                           # Elemento [Fila, Columna]

# Operaciones Avanzadas
det(matriz1)                            # Determinante
I5 <- diag(5)                           # Matriz identidad de 5x5
solve(matriz1)                          # Inversa de una matriz
round(matriz1, digits = 3)              # Redondear valores

# --- 4. Manipulación de Data Frames -------------------------------------------

# Nombres y estructura
colnames(matriz1) <- c("ingreso", "gasto", "CC")
rownames(matriz1) <- c("antara", "santa Fe", "Artz")

# Combinar datos
cbind(matriz1, x1)                      # Pegar como columna
rbind(matriz1, c(7, 4, 2))              # Pegar como fila

# Filtrado y Selección
matriz1[, -4]                           # Eliminar columna 4
matriz1[-4, ]                           # Eliminar fila 4
df$id <- 1:nrow(df)                     # Agregar contador/ID
df[2, 3:5]                              # Fila 2, columnas de la 3 a la 5
df[df$country == "TW", ]                # Filtrar por condición
df[df$country == "TW" & df$age >= 55, ] # Filtrado múltiple

# Ordenar datos
df <- df[order(df$age, decreasing = FALSE), ]

# --- 5. Estadística Descriptiva -----------------------------------------------

summary(dat)                            # Resumen estadístico general
var(dat)                                # Matriz de varianzas y covarianzas
diag(var(dat))                          # Extraer solo las varianzas
sd(dat$Variable)                        # Desviación estándar
sqrt(diag(var(dat)))                    # Desviaciones estándar de todas las variables
cor(dat$Var1, dat$Var2)                 # Correlación de Pearson
cov(dat$Var1, dat$Var2)                 # Covarianza
cv <- sd(x) / mean(x)                   # Coeficiente de variación
