# Pacotes
{
  library(haven)
  library(dplyr)
  library(naniar)
  library(ggplot2)
  library(caret)
  library(TSLSTMplus)
}

# Diretorio
#setwd("C:\\Users\\LES\\Downloads")

# Legenda
# DT_INTER : Data
# casos_resp (Variável alvo): N de internações no dia

# (Covariáveis)
# temperatura_c_media :
# pm25_ugm3_medio : material particular do fino
# umidade_relativa_percentual_media: umidade
# vento_velocidade_ms_media : Velocidade do vento

# Base de Dados Original
base = readRDS("base_resp.rds")
summary(base)
gg_miss_var(base)


# Base de Dados Treino e Teste
# Teste: 3 meses finais da base original (Set-Out-Nov)
base_treino = base |> filter(DT_INTER <= "2019-08-31") 

base_teste = base |> filter(DT_INTER > "2019-08-31")

# Definições
# Definir o Lag da Variável resposta
# Definir o Lag das Covariáveis

# Definir as escalas das covariáveis e variável resposta


### Treinamento do Modelo

# Variavel resposta e Covariavel da Base Treino
Y = base |> 
  select(casos_resp) |> as.matrix()

X = base |> 
  select(DT_INTER,temperatura_c_media,pm25_ugm3_medio,casos_resp) %>% 
  select(-c(casos_resp,DT_INTER)) |> as.matrix()

t = nrow(base_treino)
p = nrow(base) - nrow(base_treino)

# Modelo

set.seed(122054005)
lstm_sUmidade <- ts.lstm(
  ts = Y[1:t],
  xreg = X[1:t,],
  tsLag = 1,
  xregLag = 1,
  LSTMUnits = 1,
  DenseUnits = 0,
  ScaleInput = "minmax",
  ScaleOutput = "minmax",
  BatchSize = 1,
  RandomState = 122054005,
  LagsAsSequences = T)
#LSTMActivationFn = "sigmoid",
#LSTMRecurrentActivationFn ="tanh" )