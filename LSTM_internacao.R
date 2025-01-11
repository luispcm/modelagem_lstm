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
base_treino = base |> filter(DT_INTER <= "2019-08-31") |> 
  select(DT_INTER,temperatura_c_media,pm25_ugm3_medio,
         umidade_relativa_percentual_media,vento_velocidade_ms_media,casos_resp)

base_teste = base |> filter(DT_INTER > "2019-08-31")|> 
  select(DT_INTER,temperatura_c_media,pm25_ugm3_medio,
         umidade_relativa_percentual_media,vento_velocidade_ms_media,casos_resp)

# Definições
# Definir o Lag da Variável resposta
# Definir o Lag das Covariáveis

# Definir as escalas das covariáveis e variável resposta


### Treinamento do Modelo

# Variavel resposta e Covariavel da Base Treino
Y_treino = base_treino |> select(casos_resp) |> as.matrix()
X_treino = base_treino |> select(-c(casos_resp,DT_INTER)) |> as.matrix()


# Parametros do Modelo
lag = 7

# Modelo
lstm <- ts.lstm(
  ts = Y_treino,
  xreg = X_treino,
  tsLag = lag,
  xregLag = lag,
  LSTMUnits = 50, 
  ScaleInput = NULL,
  ScaleOutput = "minmax",
  LSTMActivationFn = "relu",
  LSTMRecurrentActivationFn ="sigmoid" )

### Previsão do Treino
# Previsao não interativa
prev_treino = predict(lstm,
                horizon = length(Y_treino), # Previsão para todos os períodos da base de teste
                xreg = X_treino,           # Covariáveis usadas no treino
                ts = Y_treino,             # Série temporal usada no treino
                xreg.new = X_treino)         # Covariáveis da base de teste



# Tabela para os grafico das previsões
dados_prev_treino = data.frame(indice = 1:length(Y_treino),previsao = prev_treino,Y_treino)


dados_prev_treino |> ggplot(mapping = aes(x = indice )) +
  geom_line(mapping = aes(y = previsao),color = "red") +
  geom_line(mapping = aes(y = casos_resp),color = "blue")


# Metricas
postResample(pred = previ,obs = Y_treino)















### Previsões da base Teste
Y_prev = base_teste |> select(casos_resp)|> as.matrix()
X_prev = base_teste |> select(-c(casos_resp,DT_INTER))|> as.matrix()


# Previsao não interativa
previ = predict(lstm,
        horizon = length(Y_prev), # Previsão para todos os períodos da base de teste
        xreg = X_treino,           # Covariáveis usadas no treino
        ts = Y_treino,             # Série temporal usada no treino
        xreg.new = X_prev)         # Covariáveis da base de teste



# Tabela para os grafico das previsões
dados_prev_n_int = data.frame(indice = 1:length(previ),previsao = previ,Y_prev)


dados_prev_n_int |> ggplot(mapping = aes(x = indice )) +
  geom_line(mapping = aes(y = previsao),color = "red") +
  geom_line(mapping = aes(y = casos_resp),color = "blue")


# Metricas
postResample(pred = previ,obs = Y_prev)

