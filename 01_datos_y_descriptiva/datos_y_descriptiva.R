# ============================================================================
# MÓDULO 01 — DATOS Y ESTADÍSTICA DESCRIPTIVA
# ============================================================================
# Repositorio: pronosticos_de_negocios
#
# Propósito:
# - Guía conceptual y práctica de referencia en R
# - No está diseñada para ejecutarse de principio a fin
# - Pensada como chuleta / apuntes estructurados
#
# Autor: Carlos Ocampo
# ============================================================================


# --- 1. Paquetes y Configuración ---------------------------------------------

# Paquetes comúnmente usados en econometría y pronósticos:
# tidyverse   -> manipulación y visualización de datos
# forecast    -> modelos de series de tiempo
# fpp2        -> datasets y utilidades para forecasting
# lmtest      -> pruebas estadísticas sobre modelos
# tseries     -> pruebas de estacionariedad y series de tiempo
# TSA         -> análisis clásico de series de tiempo
# MASS        -> herramientas estadísticas
#
# Nota: no se cargan explícitamente para mantener el script ligero
# library(tidyverse)

# Gestión de directorio:
# Se recomienda trabajar con proyectos de RStudio (.Rproj)
getwd()
# setwd("ruta/a/tu/proyecto")  # Evitar rutas absolutas para portabilidad


# --- 2. Manejo de Datos (Entrada / Salida) -----------------------------------

# Lectura de datos
datos_csv <- read.csv("rseIntro.csv")
datos_txt <- read.table("cableTV.dat", header = TRUE)

# Escritura de datos
write.csv(datos_csv, "cableTV_limpio.csv", row.names = FALSE)

# Conversión de tipos de datos
# Nota: `dat` es un data frame de ejemplo
dat$grupo       <- as.factor(dat$grupo)                          # Factor
dat$fecha       <- as.Date(dat$fecha, format = "%d/%m/%Y")       # Fecha
dat$texto       <- as.character(dat$texto)                       # Texto
df_matriz       <- as.data.frame(matriz1)                        # Matriz -> DF


# --- 3. Vectores, Matrices y Álgebra Lineal ----------------------------------

# Vectores
x1 <- c(7, 2, 9)
x2 <- c(1, 5, 8)

x3 <- t(x2)                      # Transpuesta
suma_vectores <- x1 + x2         # Suma elemento a elemento
escalar       <- x1 * 7          # Producto por escalar
prod_punto    <- x1 %*% x2       # Producto punto (resultado escalar)

x1[2]                            # Segundo elemento
combinado <- c(x1, x2)           # Concatenación

# Matrices
matriz1 <- matrix(
  c(1, 1, 1,
    2, 2, 2,
    4, 3, 5),
  nrow = 3,
  byrow = TRUE
)

matriz2 <- matrix(
  c(2, 0, 1,
    1, 3, 2,
    0, 1, 1),
  nrow = 3,
  byrow = TRUE
)

multiplicacion <- matriz1 %*% matriz2
matriz1[2, 3]                    # Elemento fila 2, columna 3

# Operaciones avanzadas
det(matriz1)                     # Determinante
I3 <- diag(3)                    # Matriz identidad
solve(matriz1)                   # Inversa (si existe)
round(matriz1, digits = 3)       # Redondeo


# --- 4. Manipulación de Data Frames ------------------------------------------

# Supongamos un data frame `df`
# df <- data.frame(country, age, income, gender)

# Nombres
colnames(matriz1) <- c("ingreso", "gasto", "cc")
rownames(matriz1) <- c("antara", "santa_fe", "artz")

# Combinar datos
cbind(matriz1, x1)               # Agregar columna
rbind(matriz1, c(7, 4, 2))       # Agregar fila

# Selección y filtrado (Base R)
df$id <- 1:nrow(df)              # ID incremental
df[2, 3:5]                       # Fila 2, columnas 3 a 5
df[df$country == "TW", ]
df[df$country == "TW" & df$age >= 55, ]

# Ordenar
df <- df[order(df$age, decreasing = FALSE), ]


# --- 5. Estadística Descriptiva ----------------------------------------------

# Resumen general
summary(dat)

# Varianza y covarianza
var(dat)                         # Matriz var-cov
diag(var(dat))                   # Varianzas individuales
sd(dat$variable)                 # Desviación estándar
sqrt(diag(var(dat)))             # SD de todas las variables

# Relación entre variables
cor(dat$var1, dat$var2)          # Correlación de Pearson
cov(dat$var1, dat$var2)          # Covarianza

# Coeficiente de variación
x  <- dat$variable
cv <- sd(x) / mean(x)
