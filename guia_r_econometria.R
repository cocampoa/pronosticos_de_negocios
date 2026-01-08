# ==============================================================================
# GUÍA DE REFERENCIA RÁPIDA: ESTADÍSTICA Y SERIES DE TIEMPO
# ==============================================================================


# NOTA:
# Este archivo es una guía de referencia conceptual.
# No está diseñado para ejecutarse de principio a fin.


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

# Pruebas de hipótesis para correlación
cor.test(dat$V1, dat$V2, method = "pearson")
cor.test(dat$V1, dat$V2, alternative = "greater", conf.level = 0.99)

# --- 6. Distribuciones y Probabilidad -----------------------------------------

# Distribución Normal
# qnorm: Encuentra el valor x dado una probabilidad
qnorm(0.2033, mean = 0, sd = 1, lower.tail = TRUE) 

# pnorm: Encuentra la probabilidad dada un valor x
pnorm(72, mean = 78, sd = 36, lower.tail = FALSE)

# Probabilidad Condicional P(A|B) = P(A ∩ B) / P(B)
prob_cond <- pnorm(84, 78, 36, lower.tail = F) / pnorm(72, 78, 36, lower.tail = F)

# Distribución t-student
qt(0.01, df = 52, lower.tail = FALSE)   # Valor crítico en tablas

# --- 7. Análisis de Regresión (Conceptos) -------------------------------------

# R-cuadrado (Bondad de ajuste)
r_sq <- 0.69^2                          # % variabilidad explicada por X
resto <- 1 - 0.69^2                     # % explicado por otras variables

# Estadístico de prueba manual
# Tobs <- r * sqrt(n-2) / sqrt(1 - r^2)
Tobs <- 0.36 * (sqrt(18)) / sqrt(1 - 0.36^2)

#--- 8. Visualización de Datos (Base R) --------------------------------------

# Configuración del área de trazado (2 filas, 1 columna)
par(mfrow = c(2, 1)) 

# Boxplots (Diagramas de caja)
boxplot(dat$Infant.Mortality, main = "Mortalidad", col = "hotpink", horizontal = TRUE)
boxplot(serie, main = "Producción Cerveza", col = "orange", xlab = "Megalitros")

# Histogramas y Densidad
hist(dat$Infant.Mortality, main = "Histograma", col = rainbow(5), freq = FALSE)
hist(resPez, breaks = 10, col = "lightblue") # 'breaks' controla el número de barras

# Gráficos de Dispersión y Regresión
plot(dat$Catholic, dat$Agriculture, xlab = "Agri", ylab = "Cato", main = "Relación Agri vs Cato")
abline(mod1, col = "red") # Añade la línea de regresión calculada
text(x_coord, y_coord, "y = b0 + b1x", cex = 0.8) # Añade texto al gráfico

# Gráficos de Diagnóstico del Modelo
plot(mod1)          # Genera 4 gráficos automáticos de diagnóstico (Residuos, QQ-Plot, etc.)
qqnorm(residuos)    # Gráfico de normalidad
qqline(residuos)    # Línea de referencia para normalidad

#--- 9. Modelo de Regresión Lineal Simple -----------------------------------
# Ajuste del modelo
mod1 <- lm(y ~ x, data = datos)         # Regresión lineal simple
mod2 <- lm(y ~ x1 + x2, data = datos)   # Regresión múltiple
mod_sin_intercepto <- lm(y ~ -1 + x)    # Obliga a pasar por el origen (0,0)

# Extracción de resultados
summary(mod1)                           # Resumen completo (Coeficientes, R2, p-values)
coef(mod1)                              # Extrae b0 y b1
residuals(mod1)                         # Extrae los residuales (e = y - y_ajustado)
fitted(mod1)                            # Extrae los valores ajustados (y_gorro)
anova(mod1)                             # Tabla de Análisis de Varianza (ANOVA)
vcov(mod1)                              # Matriz de varianza-covarianza de los estimadores

# Cálculo manual de métricas (Conceptos clave)
n <- length(y)
Sxx <- sum(x^2) - n * mean(x)^2
Sxy <- sum(x*y) - n * mean(x) * mean(y)
beta1 <- Sxy / Sxx                      # Pendiente
beta0 <- mean(y) - beta1 * mean(x)      # Intercepto

#--- 10. Pruebas de Hipótesis y Distribuciones -----------------------------
# Pruebas para Correlación
cor.test(dat$Var1, dat$Var2, method = "pearson")
cor.test(dat$V1, dat$V2, alternative = "greater", conf.level = 0.99)

# Valores Críticos (Tablas)
# qt, qf, qnorm se usan para encontrar el valor de corte dado un nivel de significancia (alpha)
alpha <- 0.05
gl <- n - 2
t_critico <- qt(alpha / 2, df = gl, lower.tail = FALSE) # Prueba de dos colas
f_critico <- qf(alpha, df1 = 1, df2 = gl, lower.tail = FALSE)

# P-Values (Cálculo manual del área bajo la curva)
# pt, pf, pnorm dan el área a la izquierda (o derecha si lower.tail = F)
p_value_t <- 2 * pt(t_observado, df = gl, lower.tail = FALSE)
p_value_f <- pf(f_observado, df1 = 1, df2 = gl, lower.tail = FALSE)

# Prueba de hipótesis específica para un coeficiente
# H0: beta1 = 300 vs Ha: beta1 > 300
t_est <- (coef(mod1)[2] - 300) / sqrt(vcov(mod1)[2,2])
p_val <- pt(t_est, df = gl, lower.tail = FALSE)

#--- 11. Introducción a Series de Tiempo ------------------------------------
# Declaración de objeto de serie de tiempo
serie <- ts(datos_columna, start = c(1991, 1), frequency = 12) # Mensual

# Visualización básica
plot(serie, main = "Serie de Tiempo", col = "blue", lwd = 2)
# autoplot(serie) # Requiere paquete ggplot2 + forecast

#herramientas de Diagnóstico de Modelos (Supuestos) e Intervalos de Confianza y Predicción, que son fundamentales para la estadística inferencial.He estandarizado el uso de funciones automáticas como predict() y confint(), manteniendo las fórmulas manuales como comentarios para que entiendas qué ocurre "detrás del código".Cheat Sheet de R: Inferencial y Diagnóstico
#--- 11. Intervalos de Confianza (IC) ---------------------------------------
# IC para los parámetros del modelo (b0, b1, ..., bk)
confint(mod1, level = 0.95)

# IC para la Correlación (Manual)
r <- cor(x, y)
t_critico <- qt(0.05/2, df = n-2, lower.tail = FALSE)
error_est_cor <- sqrt((1 - r^2) / (n - 2))
ic_cor <- c(r - t_critico * error_est_cor, r + t_critico * error_est_cor)

#--- 12. Estimación y Predicción --------------------------------------------
# 1. Crear nueva observación (Asegurar que los nombres de columnas coincidan)
nueva_obs <- data.frame(time = 53, price = 7, price_combo = 7, tickets = 2.2)

# 2. Intervalo de Confianza (Para la RESPUESTA MEDIA - E[Y|X])
# Responde a: "¿Cuál es el valor promedio esperado para este escenario?"
predict(mod2, newdata = nueva_obs, interval = "confidence", level = 0.95)

# 3. Intervalo de Predicción (Para una OBSERVACIÓN INDIVIDUAL)
# Responde a: "¿En qué rango caerá un dato puntual en el futuro?"
# Nota: Este intervalo siempre es más ancho que el de confianza.
predict(mod2, newdata = nueva_obs, interval = "prediction", level = 0.95)

#--- 13. Diagnóstico de Supuestos y Errores ----------------------------------
#Para que un modelo de regresión sea válido, los residuales ($e = y - \hat{y}$) deben cumplir ciertos supuestos:A. Verificación Visual (Gráficos)Rres <- residuals(mod1)    
# Extraer residuales
ajustados <- fitted(mod1) # Extraer valores estimados (y gorro)

# Homocedasticidad (Varianza constante) y Linealidad
plot(ajustados, res, main = "Residuales vs Ajustados", 
     xlab = "Valores Ajustados", ylab = "Residuales")
abline(h = 0, col = "red")

# Normalidad (QQ-Plot)
qqnorm(res)
qqline(res, col = "blue")

# Independencia (Serie de tiempo de residuales)
plot(res, type = "b", main = "Residuales a través del tiempo")
abline(h = 0)
B. Pruebas Estadísticas de NormalidadR# Prueba Jarque-Bera (Requiere paquete 'tseries')
# H0: Los datos siguen una distribución normal
library(tseries)
jarque.bera.test(res)

# Prueba de Shapiro-Wilk (Para muestras pequeñas n < 50)
shapiro.test(res)

# Estadísticos de forma (Requiere paquete 'moments')
library(moments)
skewness(res)  # Asimetría (0 = Simétrica)
kurtosis(res)  # Curtosis (3 = Normal)

#--- 14. Introducción a Series de Tiempo (Estructura) -----------------------
# Crear objeto Time Series (ts)
# frequency: 1=Anual, 4=Trimestral, 12=Mensual, 52=Semanal
miserie <- ts(datos_vector, start = c(2012, 1), frequency = 4)

# Exploración de la estructura temporal
frequency(miserie)   # Frecuencia de los datos
deltat(miserie)      # Tiempo entre observaciones (ej. 1/12 para mensual)
cycle(miserie)       # Indica la posición de cada dato en el ciclo (ej. qué mes es)

# Autocorrelación (ACF)
# Útil para detectar patrones estacionales o dependencia temporal
acf(miserie, main = "Correlograma (ACF) de la Serie")

# ==============================================================================
# GUÍA DE REFERENCIA: SERIES DE TIEMPO (PRONÓSTICOS Y DESCOMPOSICIÓN)
# ==============================================================================

# --- 15. Métodos Naive (Simples) ----------------------------------------------
# Requiere paquete: forecast
pronostico  <- naive(subserie, h = 1)      # Naive simple (próximo = actual)
pronostico2 <- snaive(subserie, h = 4)     # Naive estacional (próximo = mismo periodo anterior)

# Análisis de Autocorrelación (ACF) de residuales
Acf(res, type = "correlation", plot = TRUE)
Acf(res, type = "covariance", plot = FALSE)
# Exportar tabla de ACF a archivo local
df_acf <- data.frame(Lag = Acf(res)$lag, ACF = Acf(res)$acf)
write.table(df_acf, "resacf.dat", row.names = FALSE)

# --- 16. Índices Estacionales (Cálculo Manual por Promedios) ------------------

# 1. Obtener promedios anuales (ejemplo para meses)
prom_anuales <- sapply(seq(1, length(datos$Ventas), 12), function(i) mean(datos$Ventas[i:(i+11)]))

# 2. Obtener valores relativos (Dato / Promedio de su año)
ventas_relativas <- datos$Ventas / rep(prom_anuales, each = 12)

# 3. Calcular Índice Estacional (SI) promediando los mismos meses de cada año
# Supone 8 años de datos (96 meses)
si <- sapply(1:12, function(m) {
  indices_mes <- seq(m, length(ventas_relativas), 12)
  mean(ventas_relativas[indices_mes], na.rm = TRUE)
})

# Mostrar Índices Estacionales redondeados
round(t(si), 2)

# Desestacionalizar un valor manualmente:
# Valor_Real / Indice_Estacional

# --- 17. Medidas de Error y Precisión -----------------------------------------

# Función automática
accuracy(modelo)

# Cálculo manual selectivo (ej. para descartar periodos de ajuste inicial)
rango  <- 7:90
error  <- datos$Ventas[rango] - ypred[rango]
MAE    <- mean(abs(error))                    # Error Absoluto Medio
MSE    <- mean(error^2)                       # Error Cuadrático Medio
MAPE   <- mean(abs(error / datos$Ventas[rango])) * 100 # Error Porcentual Abs. Medio

# --- 18. Descomposición de la Serie (Y = T * S * C * I) -----------------------

# A. Método Automático
descomp <- decompose(serie, type = "multiplicative")
plot(descomp)

# B. Método Manual Paso a Paso
orden <- frequency(serie)
# 1. Tendencia mediante Promedio Móvil Centrado (CMA)
cma   <- ma(serie, order = orden, centre = TRUE)

# 2. Factor Estacional Bruto (SF)
sf    <- serie / cma

# 3. Índices Estacionales normalizados (SI)
si_raw <- sapply(1:orden, function(i) mean(sf[seq(i, length(sf), orden)], na.rm = TRUE))
si     <- si_raw * orden / sum(si_raw)  # Asegurar que sumen el orden (ej. 12)
si_serie <- rep(si, length.out = length(serie))

# 4. Tendencia (Modelo Lineal sobre CMA)
tiempo   <- 1:length(serie)
mod_tend <- lm(cma ~ tiempo, na.action = na.exclude)
cmat     <- predict(mod_tend, newdata = data.frame(tiempo = tiempo))

# 5. Factor Cíclico (CF)
cf <- cma / cmat

# 6. Recomposición (Pronóstico Y_Gorro)
ypred <- cmat * si_serie * cf
autoplot(serie) + autolayer(ts(ypred, start = start(serie), freq = orden))

# --- 19. Diferenciación y Estacionariedad -------------------------------------

# Determinar número de diferencias necesarias
ndiffs(serie)   # Diferencias para quitar tendencia (d)
nsdiffs(serie)  # Diferencias para quitar estacionalidad (D)

# Aplicar transformaciones
plot(log(serie))            # Estabilizar varianza
plot(diff(log(serie)))      # Serie diferenciada de orden 1 (quitar tendencia)
plot(diff(diff(log(serie)), lag = 12)) # Diferencia estacional (orden 12)

# Límites significativos del Autocorrelograma (Cálculo manual)
n_obs    <- length(serie)
z_tablas <- qnorm(0.05/2, lower.tail = FALSE)
limite_inf <- -z_tablas / sqrt(n_obs)
limite_sup <-  z_tablas / sqrt(n_obs)

# --- 20. Pronóstico con Intervalos (Descomposición) ---------------------------

# Ejemplo para un tiempo 't' futuro:
t_futuro <- 97 
b0 <- coef(mod_tend)[1]
b1 <- coef(mod_tend)[2]

Tt <- b0 + b1 * t_futuro    # Tendencia proyectada
St <- si[1]                 # Índice del mes correspondiente
Ct <- 1.0                   # Factor de ciclo (asumiendo 1 si no hay info)

Y_pronos <- Tt * St * Ct

# Intervalos de Confianza para el Pronóstico
m     <- n_obs - orden      # Grados de libertad ajustados
t_tab <- qt(0.05/2, m - 1, lower.tail = FALSE)
eee   <- sqrt(sum(error^2) / (m - 1)) # Error estándar de los errores

LI <- Y_pronos - t_tab * eee
LS <- Y_pronos + t_tab * eee

Aquí tienes el bloque integrado de Regresión Lineal Múltiple. He limpiado el código para que sea una herramienta de cálculo matricial (útil para entender la teoría) y también incluya los comandos rápidos de R para la práctica profesional.

Todo se mantiene en un solo bloque para tu facilidad:

R

# ==============================================================================
# GUÍA DE REFERENCIA: REGRESIÓN LINEAL MÚLTIPLE Y ÁLGEBRA MATRICIAL
# ==============================================================================

# --- 21. Construcción Matricial del Modelo (Cálculo Manual) -------------------
# Modelo: y = X*beta + error

# 1. Preparación de matrices
y <- Influencersc$SALARY
n <- length(y)
# Añadir columna de 1s para la ordenada al origen (Intercepto)
X <- cbind(1, Influencersc$TIME, Influencersc$PUB, Influencersc$CIT1, Influencersc$CITM)
colnames(X) <- c("b0", "TIME", "PUB", "CIT1", "CITM")

# 2. Estimación de Coeficientes: beta_gorro = (X'X)^-1 * X'y
XtX <- t(X) %*% X
InvXtX <- solve(XtX)
bgorro <- InvXtX %*% t(X) %*% y

# 3. Valores ajustados y Residuales
ygorro <- X %*% bgorro
res <- y - ygorro

# 4. Varianza de los errores (s2) y de los estimadores
q <- ncol(X) # Número de parámetros
SCres <- as.numeric(t(res) %*% res)
s2 <- SCres / (n - q)
error_estandar_reg <- sqrt(s2)

varcov_bgorro <- s2 * InvXtX           # Matriz de varianza-covarianza
ee_bgorro <- sqrt(diag(varcov_bgorro)) # Errores estándar de cada beta

# --- 22. Análisis de Bondad de Ajuste (Métricas) ------------------------------
bary <- mean(y)
SCtotal <- sum((y - bary)^2)
SCreg <- SCtotal - SCres

R2 <- SCreg / SCtotal
R2ajus <- 1 - ((1 - R2) * (n - 1) / (n - q))

# Estadístico F (Prueba global del modelo)
CMreg <- SCreg / (q - 1)
CMres <- s2
Fobs <- CMreg / CMres
valor_p_F <- pf(Fobs, q - 1, n - q, lower.tail = FALSE)

# --- 23. Pruebas de Hipótesis Individuales (t-test) ---------------------------
# H0: betaj = 0
j <- 2 # Índice del parámetro a probar
Tjobs <- bgorro[j] / ee_bgorro[j]
valor_p_t <- 2 * pt(abs(Tjobs), n - q, lower.tail = FALSE)

# --- 24. Intervalos de Confianza y Predicción Matricial -----------------------
# Para un nuevo vector de características x0:
x0 <- c(1, 1, 110, 15, 200) 
alpha <- 0.05
t_conf <- qt(alpha/2, n - q, lower.tail = FALSE)

# A. Intervalo de Confianza (Respuesta media)
se_ygorro <- sqrt(s2 * (t(x0) %*% InvXtX %*% x0))
IC_matriz <- c(t(x0) %*% bgorro - t_conf * se_ygorro, t(x0) %*% bgorro + t_conf * se_ygorro)

# B. Intervalo de Predicción (Nueva observación individual)
se_ypunto <- sqrt(s2 * (1 + t(x0) %*% InvXtX %*% x0))
IP_matriz <- c(t(x0) %*% bgorro - t_conf * se_ypunto, t(x0) %*% bgorro + t_conf * se_ypunto)

# --- 25. Herramientas de Regresión con Funciones de R -------------------------

# Ajuste rápido
modelo_mult <- lm(SALARY ~ TIME + PUB + CIT1 + CITM, data = Influencersc)
summary(modelo_mult)

# Comparación de modelos (ANOVA)
anova(modelo_mult)

# Análisis descriptivo por categorías (Variables Dummy/Factores)
# Promedio de salario cruzando Sexo y estatus VIP
aggregate(SALARY ~ SEX + VIP, data = Influencersc, mean)
table(Influencersc$SEX, Influencersc$VIP) # Frecuencias

# Diagnóstico Rápido: ¿T^2 = F? (Solo en Regresión Simple)
# En regresión simple, el estadístico t del predictor al cuadrado es igual al F de la tabla ANOVA.
# t_val <- summary(mod)$coefficients[2,3]; f_val <- summary(mod)$fstatistic[1]
# t_val^2 == f_val 

# Predicciones con funciones nativas
nueva_obs <- data.frame(TIME=1, PUB=110, CIT1=15, CITM=200)
pred_conf <- predict(modelo_mult, newdata = nueva_obs, interval = "confidence")
pred_pred <- predict(modelo_mult, newdata = nueva_obs, interval = "prediction")

# Visualización de Diagnóstico
plot(modelo_mult) # Genera los 4 gráficos de supuestos


# ==============================================================================
# GUÍA DE REFERENCIA: REGRESIÓN MÚLTIPLE, INTERACCIONES Y DIAGNÓSTICO
# ==============================================================================

# --- 26. Regresión Múltiple e Interpretación ----------------------------------
# Modelo con múltiples predictores
inf2 <- lm(SALARY ~ PUB + TIME, data = influencers)
summary(inf2)

# Interpretación rápida:
# Intercepto (b0): Valor esperado de Y cuando todos los X son 0.
# Coeficientes (bj): Cambio promedio en Y por unidad de Xj, manteniendo lo demás constante.

# --- 27. Pruebas de Hipótesis Específicas (Unilateral) ------------------------
# Ejemplo: Probar si el efecto de las publicaciones es menor a 200
# H0: beta_pub >= 200 vs Ha: beta_pub < 200
beta_H0 <- 200
b_gorro <- coef(inf2)["PUB"]
ee_b    <- summary(inf2)$coefficients["PUB", "Std. Error"]
n <- nrow(influencers)
p <- 2 # número de predictores (sin contar intercepto)

t_est <- (b_gorro - beta_H0) / ee_b
p_val <- pt(t_est, df = n - (p + 1), lower.tail = TRUE)

# Valor Crítico (Corte para b_pub):
# b_critico = t_tablas * ee_b + beta_H0
t_tablas <- qt(0.05, df = n - (p + 1), lower.tail = TRUE)
valor_critico_b <- t_tablas * ee_b + beta_H0

# --- 28. Interacciones y Estandarización --------------------------------------
# Modelo con interacción: el efecto de una variable depende del nivel de la otra
# Y = b0 + b1*X1 + b2*X2 + b3*(X1*X2)
infInt <- lm(SALARY ~ PUB + TIME + PUB:TIME, data = influencers)
summary(infInt)

# Comparar importancia de variables (Coeficientes Estandarizados Beta)
# Requiere: install.packages("lm.beta")
library(lm.beta)
coef(lm.beta(inf2), standardized = TRUE) 

# Modelo sin intercepto (La R2 cambia su significado, usar con precaución)
inf_sin_b0 <- lm(SALARY ~ -1 + PUB + TIME, data = influencers)

# --- 29. Manejo de Variables Categóricas y Dummies ----------------------------
# R convierte automáticamente factores en variables Dummy (0, 1)
datos$Color <- as.factor(datos$Color)
datos$Claridad <- as.factor(datos$Claridad)

# Modelo con todas las variables (Precio en función de todas las demás)
# El "." indica "todas las variables del dataframe"
mod_completo <- lm(Precio ~ . , data = datos)

# --- 30. Detección y Limpieza de Valores Atípicos (Outliers) ------------------
# Identificación mediante Rango Intercuartílico (IQR)
# Regla: Q3 + 1.5 * IQR  o  Q1 - 1.5 * IQR
Q3 <- quantile(datos$Quilates, 0.75)
IQR_val <- IQR(datos$Quilates)
limite_superior <- Q3 + 1.5 * IQR_val

# Encontrar índices de los atípicos
indices_outliers <- which(datos$Quilates > limite_superior)

# Crear base de datos limpia (quitando filas por índice)
datos_limpios <- datos[-indices_outliers, ]

# --- 31. Intervalos de Confianza para Parámetros ------------------------------
# IC para los coeficientes al 95%
ic_coefs <- confint(mod_completo, level = 0.95)

# Ejemplo: Incremento mínimo esperado en el precio por cada medio quilate adicional
# Tomar el límite inferior de la pendiente (b1) y dividir entre 2
incremento_minimo_medio_quilate <- ic_coefs["Quilates", 1] / 2

# --- 32. Comparación de Modelos (ANOVA para modelos anidados) -----------------
# ¿Valen la pena los predictores adicionales?
# H0: Los modelos son equivalentes (el modelo simple es suficiente)
anova(mod_completo, mod_reducido)

Aquí tienes el último bloque integrado. He incluido las técnicas de predicción con variables categóricas, la creación de variables indicadoras (factores lógicos) y la prueba de diferencia de coeficientes, que es fundamental para comparar el impacto de dos predictores distintos.

Todo consolidado en un solo bloque para tu guía:

R

# ==============================================================================
# GUÍA DE REFERENCIA: PREDICCIÓN, VARIABLES INDICADORAS Y COMPARACIÓN DE BETAS
# ==============================================================================

# --- 33. Predicción Avanzada con Factores -------------------------------------
# 1. Crear el escenario (asegurar que los niveles del factor existan en la base original)
nueva_obs <- data.frame(ID = 0, Quilates = 2, Claridad = "Very Good", Color = "E")
rownames(nueva_obs) <- c("Diamante_Objetivo")

# 2. Estimación por Intervalo (90% de confianza)
# Intervalo para la RESPUESTA MEDIA (Confidence)
pred_promedio <- predict(modd1, newdata = nueva_obs, interval = "confidence", level = 0.90)

# 3. Intervalo para un DATO INDIVIDUAL (Prediction)
pred_individual <- predict(modd1, newdata = nueva_obs, interval = "prediction", level = 0.90)

# --- 34. Cálculo de Error Estándar a partir de Intervalos ---------------------
# Si tienes el límite superior (LS) y el ajuste (fit), puedes despejar el Error Estándar (EE)
# Formula: LS = fit + t_tablas * EE  =>  EE = (LS - fit) / t_tablas
t_tablas <- qt(1 - (0.1/2), modd1$df.residual)
error_est_manual <- (pred_promedio[,"upr"] - pred_promedio[,"fit"]) / t_tablas

# --- 35. Variables Lógicas e Indicadoras (Dummy Automático) -------------------
# Crear factores sobre la marcha para segmentar la base (ej. mtcars)
coches <- mtcars
coches$am <- as.factor(coches$am) # 0=Auto, 1=Manual

# Crear variable lógica: ¿Tiene más de 6 cilindros? (TRUE/FALSE)
coches$rendimiento <- factor(coches$cyl > 6)
mod_dummy <- lm(qsec ~ rendimiento, data = coches) # Compara promedios entre grupos

# Comparaciones específicas (TRUE si tiene exactamente 3 velocidades)
coches$tresvel <- factor(coches$gear == 3)
summary(lm(mpg ~ tresvel, data = coches))

# --- 36. Pruebas de Hipótesis para Coeficientes (Casos Especiales) ------------
# Prueba de correlación con dirección (Menor que / Unilateral)
cor.test(datos$tickets, datos$price, method = "pearson", alternative = "less", conf.level = 0.99)

# Prueba de Hipótesis para un Beta específico (H0: Beta_j = Valor_Hipotetico)
# Ejemplo: Probar si el impacto del precio de combo es igual a 7000
b_j <- coef(mod1)["price_combo"]
ee_j <- summary(mod1)$coefficients["price_combo", "Std. Error"]
t_obs <- (b_j - 7000) / ee_j

# --- 37. Prueba de Diferencia entre dos Coeficientes (B1 = B2) ----------------
# Útil para saber si el impacto de X1 es significativamente distinto al de X2
# Estadístico: t = [(b1 - b2) - (B1 - B2)] / sqrt(var(b1) + var(b2) - 2*cov(b1, b2))

n <- nrow(Cinema)
v_matriz <- vcov(mod1) # Matriz de varianza-covarianza

var_b1 <- v_matriz[2, 2]     # Varianza de Beta 1
var_b2 <- v_matriz[3, 3]     # Varianza de Beta 2
cov_b1b2 <- v_matriz[3, 2]   # Covarianza entre Beta 1 y Beta 2

# Diferencia observada (H0: B1 - B2 = 0)
diff_betas <- coef(mod1)[2] - coef(mod1)[3]
error_est_diff <- sqrt(var_b1 + var_b2 - 2 * cov_b1b2)

t_est_diff <- diff_betas / error_est_diff
p_val_diff <- 2 * pt(abs(t_est_diff), df = n - 4, lower.tail = FALSE)

Esta es la última pieza de tu guía. He integrado los conceptos finales de Descomposición Manual de Series de Tiempo, Pronóstico con componentes (T, S, C) y el Diagnóstico de errores (Jarque-Bera).

Con este bloque, tu guía cubre el ciclo completo: desde la carga de datos hasta la validación de modelos avanzados de series de tiempo.

R

# ==============================================================================
# GUÍA DE REFERENCIA: PRONÓSTICO AVANZADO Y DESCOMPOSICIÓN DE SERIES
# ==============================================================================

# --- 38. Descomposición de Series de Tiempo (Multiplicativa) ------------------
# Declaración de la serie (Frecuencia 4 = Trimestral)
serie <- ts(datosC$VENTAS, start = c(2017, 4), frequency = 4)

# A. Método Automático
serieD <- decompose(serie, type = "multiplicative")
accuracy(serieD$x) # Evaluación de precisión

# B. Método Manual (Paso a Paso)
orden <- frequency(serie)
# 1. Promedio Móvil Centrado (CMA) para capturar la tendencia
cma <- ma(serie, order = orden, centre = TRUE)

# 2. Factor Estacional Bruto (SF) e Índices Estacionales (SI)
sf <- serie / cma
si_raw <- sapply(1:orden, function(i) mean(sf[seq(i, length(sf), orden)], na.rm = TRUE))
si <- si_raw * orden / sum(si_raw) # Normalización: los índices deben sumar 'orden'

# 3. Componente de Tendencia (Lineal sobre CMA)
tiempo <- 1:length(serie)
cma_mod <- lm(cma ~ tiempo, na.action = na.exclude)
cmat <- fitted(cma_mod)

# 4. Factor Cíclico (CF)
cf <- cma / cmat

# 5. Recomposición y Pronóstico (Y_gorro = T * S * C)
si_serie <- rep(si, length.out = length(serie))
ypred <- cmat * si_serie * cf # Valores ajustados por descomposición

# Visualización del ajuste
autoplot(serie) + autolayer(ts(ypred, start = start(serie), freq = orden))

# --- 39. Pronóstico para Periodos Futuros (t + h) -----------------------------
# Supongamos que queremos predecir el periodo t = 19
nueva_t <- data.frame(tiempo = 19)
pred_tendencia <- predict(cma_mod, newdata = nueva_t, interval = "confidence")

fit_T <- pred_tendencia[1]  # Valor de la tendencia proyectada
S_i   <- si[4]              # Factor estacional del trimestre 4
C_prom <- mean(cf, na.rm=T) # Usamos el ciclo promedio

# Pronóstico puntual
y_punto <- C_prom * S_i * fit_T

# --- 40. Intervalos de Confianza para Pronósticos de Series -------------------
m  <- length(na.omit(ypred)) # Datos efectivos usados
gl <- m - 1
t_tablas <- qt(0.05/2, gl, lower.tail = FALSE)

# Error Estándar de la Estimación (EEE)
error2 <- datosC$VENTAS[3:16] - ypred[3:16] # Ajustar según datos disponibles
eee <- sqrt(sum(error2^2) / gl)

# Límites del Intervalo
LI <- y_punto - t_tablas * eee
LS <- y_punto + t_tablas * eee

# --- 41. Diagnóstico de Normalidad y Bondad de Ajuste -------------------------
# Prueba Jarque-Bera (H0: Los errores son normales)
# Requiere: library(tseries)
res_modelo <- residuals(lm(VENTAS ~ PROMCOM, data = datosC))
jarque.bera.test(res_modelo)

# Nivel de confianza máximo (1 - p-value)
# Si el p-value es 0.146, el nivel de confianza máximo para no rechazar es 85.4%
conf_max <- 1 - 0.146

# --- 42. Cálculo Manual de F-obs y Valor P ------------------------------------
n_obs <- 18
SCres <- 80336573.82
SCreg <- 57174250.56

CMreg <- SCreg / 1
CMres <- SCres / (n_obs - 2)
F_obs <- CMreg / CMres

# Valor p de la prueba F (¿Es significativo el modelo?)
p_val_f <- pf(F_obs, 1, n_obs - 2, lower.tail = FALSE)

# --- 43. Límites Críticos para Autocorrelación (ACF) -------------------------
# Para determinar si un rezago es estadísticamente significativo
n_total <- length(serie)
z_critico <- qnorm(0.05/2, lower.tail = FALSE)
LI_acf <- -z_critico / sqrt(n_total)
LS_acf <-  z_critico / sqrt(n_total)

# ==============================================================================
# LICENCIA MIT
# Copyright (c) 2024 Carlos Ocampo
# Se concede permiso para usar, copiar y modificar este código con fines 
# educativos y profesionales, siempre que se mantenga este aviso de crédito.
# ==============================================================================















