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


# Parametros do Modelo
lag = 3

# Modelo
set.seed(122054005)
lstm_sUmidade <- ts.lstm(
  ts = Y[1:t],
  xreg = X[1:t,],
  tsLag = 0,
  xregLag = lag,
  LSTMUnits = c(40,20), 
  ScaleInput = "minmax",
  ScaleOutput = "minmax",RandomState = 122054005)
  #LSTMActivationFn = "sigmoid",
  #LSTMRecurrentActivationFn ="tanh" )

### Previsão do Treino
# Previsao não interativa
prev_treino = predict(lstm_sUmidade,
                      horizon = t, # Previsão para todos os períodos da base de teste
                      xreg = X[1:t,],           # Covariáveis usadas no treino
                      ts = Y[1:t],             # Série temporal usada no treino
                      xreg.new = X[1:t,])         # Covariáveis da base de teste



# Tabela para os grafico das previsões
dados_prev_treino = data.frame(data = base$DT_INTER[1:t],previsao = prev_treino,Internacao = Y[1:t])

g_treino_sUmidade = dados_prev_treino |> ggplot(mapping = aes(x = data )) +
  geom_line(mapping = aes(y = previsao),color = "red") +
  geom_line(mapping = aes(y = Internacao),color = "black") 
#+ geom_line(mapping = aes(y = pred_int_treino),color = "blue")

# Metricas
m1_t = postResample(pred = dados_prev_treino$previsao,obs = Y[1:t])



### Previsões da base Teste

#Previsao não interativa
previ_sUmidade = predict(lstm_sUmidade,
               horizon = p, # Previsão para todos os períodos da base de teste
               xreg = X[1:t,],           # Covariáveis usadas no treino
               ts = Y[1:t],             # Série temporal usada no treino
               xreg.new = X[(t+1):(t+p),])         # Covariáveis da base de teste


# Previsão interativa
pred_int_sUmidade = NULL
for(i in (t+1):(t+p)){
  pred_int_sUmidade = c(pred_int_sUmidade,
               predict(lstm_sUmidade, 
                       horizon = 1, 
                       xreg = X[1:(i-1),], 
                       ts = Y[1:(i-1)],
                       xreg.new = X[i, , drop = F]))
}


# Tabela para os grafico das previsões
dados_prev_sUmidade = data.frame(data = base$DT_INTER[(t+1):(t+p)],previsao = previ_sUmidade,Internacao = Y[(t+1):(t+p)],
                              previsao_int = pred_int_sUmidade)

g_LSTM_sUmidade = dados_prev_sUmidade |> ggplot(mapping = aes(x = data )) +
  geom_line(mapping = aes(y = Internacao,color = "Internação")) + 
  geom_line(mapping = aes(y = previsao_int,color = "Previsão Interativa"))+  # Previsão interativa
  geom_line(mapping = aes(y = previsao,color = "Previsão Não Interativa")) + # Previsão n inter 
  scale_color_manual(values = c("Internação" = "black", 
                                "Previsão Interativa" = "blue", 
                                "Previsão Não Interativa" = "red")) +
  labs(color = "Legenda", y = "Valores", x = "Data",
       title = "lstm s/Umidade") +
  theme_minimal()

postResample(pred = dados_prev_sUmidade$previsao,obs = Y[(t+1):(t+p)])
m1 = postResample(pred = dados_prev_sUmidade$previsao_int,obs = Y[(t+1):(t+p)])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Completo
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Variavel resposta e Covariavel da Base Treino
Y = base |> 
  select(casos_resp) |> as.matrix()

X = base |> 
  select(DT_INTER,temperatura_c_media,pm25_ugm3_medio,casos_resp,umidade_relativa_percentual_media) %>% 
  select(-c(casos_resp,DT_INTER)) |> as.matrix()

t = nrow(base_treino)
p = nrow(base) - nrow(base_treino)


# Parametros do Modelo
lag = 3

# Modelo
set.seed(122054005)
lstm_completo <- ts.lstm(
  ts = Y[1:t],
  xreg = X[1:t,],
  tsLag = lag,
  xregLag = lag,
  LSTMUnits = c(40,20), 
  ScaleInput = "minmax",
  ScaleOutput = "minmax",RandomState = 122054005)
#LSTMActivationFn = "sigmoid",
#LSTMRecurrentActivationFn ="tanh" )

### Previsão do Treino
# Previsao não interativa
prev_treino_completo = predict(lstm_completo,
                      horizon = t, # Previsão para todos os períodos da base de teste
                      xreg = X[1:t,],           # Covariáveis usadas no treino
                      ts = Y[1:t],             # Série temporal usada no treino
                      xreg.new = X[1:t,])         # Covariáveis da base de teste



# Tabela para os grafico das previsões
dados_prev_treino = data.frame(data = base$DT_INTER[1:t],previsao = prev_treino,Internacao = Y[1:t])

g_treino_completo = dados_prev_treino |> ggplot(mapping = aes(x = data )) +
  geom_line(mapping = aes(y = previsao),color = "red") +
  geom_line(mapping = aes(y = Internacao),color = "black") 
#+ geom_line(mapping = aes(y = pred_int_treino),color = "blue")

# Metricas
m2_t = postResample(pred = dados_prev_treino$previsao,obs = Y[1:t])



### Previsões da base Teste

#Previsao não interativa
previ_completo = predict(lstm_completo,
                horizon = p, # Previsão para todos os períodos da base de teste
                xreg = X[1:t,],           # Covariáveis usadas no treino
                ts = Y[1:t],             # Série temporal usada no treino
                xreg.new = X[(t+1):(t+p),])         # Covariáveis da base de teste


# Previsão interativa
pred_int_completo = NULL
for(i in (t+1):(t+p)){
  pred_int_completo = c(pred_int_completo,
               predict(lstm_completo, 
                       horizon = 1, 
                       xreg = X[1:(i-1),], 
                       ts = Y[1:(i-1)],
                       xreg.new = X[i, , drop = F]))
}


# Tabela para os grafico das previsões
dados_prev_completo = data.frame(data = base$DT_INTER[(t+1):(t+p)],previsao = previ_completo,Internacao = Y[(t+1):(t+p)],
                                 previsao_int = pred_int_completo)

g_LSTM_completo = dados_prev_completo |> ggplot(mapping = aes(x = data )) +
  geom_line(mapping = aes(y = Internacao,color = "Internação")) + 
  geom_line(mapping = aes(y = previsao_int,color = "Previsão Interativa"))+  # Previsão interativa
  geom_line(mapping = aes(y = previsao,color = "Previsão Não Interativa")) + # Previsão n inter 
  scale_color_manual(values = c("Internação" = "black", 
                                "Previsão Interativa" = "blue", 
                                "Previsão Não Interativa" = "red")) +
  labs(color = "Legenda", y = "Valores", x = "Data",
      title = "lstm completo") +
  theme_minimal()

postResample(pred = dados_prev_completo$previsao,obs = Y[(t+1):(t+p)])
m2 = postResample(pred = dados_prev_completo$previsao_int,obs = Y[(t+1):(t+p)])



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Teste de lag = 7
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Variavel resposta e Covariavel da Base Treino
Y = base |> 
  select(casos_resp) |> as.matrix()

X = base |> 
  select(DT_INTER,temperatura_c_media,pm25_ugm3_medio,casos_resp) %>% 
  select(-c(casos_resp,DT_INTER)) |> as.matrix()

t = nrow(base_treino)
p = nrow(base) - nrow(base_treino)


# Parametros do Modelo
lag = 7

# Modelo
set.seed(122054005)
lstm_lag7 <- ts.lstm(
  ts = Y[1:t],
  xreg = X[1:t,],
  tsLag = 7,
  xregLag = 7,
  LSTMUnits = c(40,20), 
  ScaleInput = "minmax",
  ScaleOutput = "minmax",RandomState = 122054005)
#LSTMActivationFn = "sigmoid",
#LSTMRecurrentActivationFn ="tanh" )

### Previsão do Treino
# Previsao não interativa
prev_treino_lag7 = predict(lstm_lag7,
                               horizon = t, # Previsão para todos os períodos da base de teste
                               xreg = X[1:t,],           # Covariáveis usadas no treino
                               ts = Y[1:t],             # Série temporal usada no treino
                               xreg.new = X[1:t,])         # Covariáveis da base de teste



# Tabela para os grafico das previsões
dados_prev_lag7 = data.frame(data = base$DT_INTER[1:t],previsao = prev_treino_lag7,Internacao = Y[1:t])

g_treino_lag7 = dados_prev_lag7 |> ggplot(mapping = aes(x = data )) +
  geom_line(mapping = aes(y = previsao),color = "red") +
  geom_line(mapping = aes(y = Internacao),color = "black") 
#+ geom_line(mapping = aes(y = pred_int_treino),color = "blue")

# Metricas
m3_t = postResample(pred = dados_prev_lag7$previsao,obs = Y[1:t])



### Previsões da base Teste

#Previsao não interativa
previ_lag7 = predict(lstm_lag7,
                         horizon = p, # Previsão para todos os períodos da base de teste
                         xreg = X[1:t,],           # Covariáveis usadas no treino
                         ts = Y[1:t],             # Série temporal usada no treino
                         xreg.new = X[(t+1):(t+p),])         # Covariáveis da base de teste


# Previsão interativa
pred_int_lag7 = NULL
for(i in (t+1):(t+p)){
  pred_int_lag7 = c(pred_int_lag7,
                        predict(lstm_lag7, 
                                horizon = 1, 
                                xreg = X[1:(i-1),], 
                                ts = Y[1:(i-1)],
                                xreg.new = X[i, , drop = F]))
}


# Tabela para os grafico das previsões
dados_prev_lag7 = data.frame(data = base$DT_INTER[(t+1):(t+p)],previsao = previ_lag7,Internacao = Y[(t+1):(t+p)],
                                 previsao_int = pred_int_lag7)

g_LSTM_lag7 = dados_prev_lag7 |> ggplot(mapping = aes(x = data )) +
  geom_line(mapping = aes(y = Internacao,color = "Internação")) + 
  geom_line(mapping = aes(y = previsao_int,color = "Previsão Interativa"))+  # Previsão interativa
  geom_line(mapping = aes(y = previsao,color = "Previsão Não Interativa")) + # Previsão n inter 
  scale_color_manual(values = c("Internação" = "black", 
                                "Previsão Interativa" = "blue", 
                                "Previsão Não Interativa" = "red")) +
  labs(color = "Legenda", y = "Valores", x = "Data",
       title = "lstm lag 7") +
  theme_minimal()

postResample(pred = dados_prev_lag7$previsao,obs = Y[(t+1):(t+p)])
m3 = postResample(pred = dados_prev_lag7$previsao_int,obs = Y[(t+1):(t+p)])



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Teste de lag = 1
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Variavel resposta e Covariavel da Base Treino
Y = base |> 
  select(casos_resp) |> as.matrix()

X = base |> 
  select(DT_INTER,temperatura_c_media,pm25_ugm3_medio,casos_resp) %>% 
  select(-c(casos_resp,DT_INTER)) |> as.matrix()

t = nrow(base_treino)
p = nrow(base) - nrow(base_treino)


# Parametros do Modelo
lag = 1

# Modelo
set.seed(122054005)
lstm_lag1 <- ts.lstm(
  ts = Y[1:t],
  xreg = X[1:t,],
  tsLag = 1,
  xregLag = 1,
  LSTMUnits = c(40,20), 
  ScaleInput = "minmax",
  ScaleOutput = "minmax",RandomState = 122054005)
#LSTMActivationFn = "sigmoid",
#LSTMRecurrentActivationFn ="tanh" )

### Previsão do Treino
# Previsao não interativa
prev_treino_lag1 = predict(lstm_lag1,
                           horizon = t, # Previsão para todos os períodos da base de teste
                           xreg = X[1:t,],           # Covariáveis usadas no treino
                           ts = Y[1:t],             # Série temporal usada no treino
                           xreg.new = X[1:t,])         # Covariáveis da base de teste



# Tabela para os grafico das previsões
dados_prev_lag1 = data.frame(data = base$DT_INTER[1:t],previsao = prev_treino_lag1,Internacao = Y[1:t])

g_treino_lag1 = dados_prev_lag1 |> ggplot(mapping = aes(x = data )) +
  geom_line(mapping = aes(y = previsao),color = "red") +
  geom_line(mapping = aes(y = Internacao),color = "black") 
#+ geom_line(mapping = aes(y = pred_int_treino),color = "blue")

# Metricas
m4_t = postResample(pred = dados_prev_lag1$previsao,obs = Y[1:t])



### Previsões da base Teste

#Previsao não interativa
previ_lag1 = predict(lstm_lag1,
                     horizon = p, # Previsão para todos os períodos da base de teste
                     xreg = X[1:t,],           # Covariáveis usadas no treino
                     ts = Y[1:t],             # Série temporal usada no treino
                     xreg.new = X[(t+1):(t+p),])         # Covariáveis da base de teste


# Previsão interativa
pred_int_lag1 = NULL
for(i in (t+1):(t+p)){
  pred_int_lag1 = c(pred_int_lag1,
                    predict(lstm_lag1, 
                            horizon = 1, 
                            xreg = X[1:(i-1),], 
                            ts = Y[1:(i-1)],
                            xreg.new = X[i, , drop = F]))
}


# Tabela para os grafico das previsões
dados_prev_lag1 = data.frame(data = base$DT_INTER[(t+1):(t+p)],previsao = previ_lag1,Internacao = Y[(t+1):(t+p)],
                             previsao_int = pred_int_lag1)

g_LSTM_lag1 = dados_prev_lag1 |> ggplot(mapping = aes(x = data )) +
  geom_line(mapping = aes(y = Internacao,color = "Internação")) + 
  geom_line(mapping = aes(y = previsao_int,color = "Previsão Interativa"))+  # Previsão interativa
  geom_line(mapping = aes(y = previsao,color = "Previsão Não Interativa")) + # Previsão n inter 
  scale_color_manual(values = c("Internação" = "black", 
                                "Previsão Interativa" = "blue", 
                                "Previsão Não Interativa" = "red")) +
  labs(color = "Legenda", y = "Valores", x = "Data",
       title = "lstm lag 1") +
  theme_minimal()

postResample(pred = dados_prev_lag1$previsao,obs = Y[(t+1):(t+p)])
m4 = postResample(pred = dados_prev_lag1$previsao_int,obs = Y[(t+1):(t+p)])



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Resultados
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# metrica do treino
metricas_t = cbind(m1_t,m2_t,m3_t,m4_t)
metricas_t


# Metrica do teste
metrica_completo = cbind(m1,m2,m3,m4)
metrica_completo

# Graficos
g_LSTM_sUmidade
g_LSTM_completo
g_LSTM_lag7
g_LSTM_lag1
