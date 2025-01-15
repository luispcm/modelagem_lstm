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
lstm <- ts.lstm(
  ts = Y[1:t],
  xreg = X[1:t,],
  tsLag = lag,
  xregLag = lag,
  LSTMUnits = c(40,20), 
  ScaleInput = "minmax",
  ScaleOutput = "minmax")
  #LSTMActivationFn = "sigmoid",
  #LSTMRecurrentActivationFn ="tanh" )

### Previsão do Treino
# Previsao não interativa
prev_treino = predict(lstm,
                      horizon = t, # Previsão para todos os períodos da base de teste
                      xreg = X[1:t,],           # Covariáveis usadas no treino
                      ts = Y[1:t],             # Série temporal usada no treino
                      xreg.new = X[1:t,])         # Covariáveis da base de teste



# Tabela para os grafico das previsões
dados_prev_treino = data.frame(data = base$DT_INTER[1:t],previsao = prev_treino,Internacao = Y[1:t])

dados_prev_treino |> ggplot(mapping = aes(x = data )) +
  geom_line(mapping = aes(y = previsao),color = "red") +
  geom_line(mapping = aes(y = Internacao),color = "black") 
#+ geom_line(mapping = aes(y = pred_int_treino),color = "blue")

# Metricas
postResample(pred = dados_prev_treino$previsao,obs = Y[1:t])



### Previsões da base Teste

#Previsao não interativa
previ = predict(lstm,
               horizon = p, # Previsão para todos os períodos da base de teste
               xreg = X[1:t,],           # Covariáveis usadas no treino
               ts = Y[1:t],             # Série temporal usada no treino
               xreg.new = X[(t+1):(t+p),])         # Covariáveis da base de teste



# Tabela para os grafico das previsões
dados_prev_n_int = data.frame(data = base$DT_INTER[(t+1):(t+p)],previsao = previ,Internacao = Y[(t+1):(t+p)])


# Previsão interativa
pred_int = NULL
for(i in (t+1):(t+p)){
  pred_int = c(pred_int,
               predict(lstm, 
                       horizon = 1, 
                       xreg = X[1:(i-1),], 
                       ts = Y[1:(i-1)],
                       xreg.new = X[i, , drop = F]))
}

dados_prev_n_int |> ggplot(mapping = aes(x = data )) +
  geom_line(mapping = aes(y = Internacao,color = "Internação")) + 
  geom_line(mapping = aes(y = pred_int,color = "Previsão Interativa"))+  # Previsão interativa
  geom_line(mapping = aes(y = previsao,color = "Previsão Não Interativa")) + # Previsão n inter 
  scale_color_manual(values = c("Internação" = "black", 
                                "Previsão Interativa" = "blue", 
                                "Previsão Não Interativa" = "red")) +
  labs(color = "Legenda", y = "Valores", x = "Data") +
  theme_minimal()
postResample(pred = dados_prev_n_int$previsao,obs = Y[(t+1):(t+p)])



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Parametros do Modelo
lag = seq(5,12)
neuron = seq(10,80,10)

# Modelo
for (l in 1:length(lag)) {
  for (n in 1:length(neuron)) {
    set.seed(122054005)
    lstm <- ts.lstm(
      ts = Y[1:t],
      xreg = X[1:t,],
      tsLag = l,
      xregLag = l,
      LSTMUnits = n, 
      ScaleInput = NULL,
      ScaleOutput = "minmax",
      LSTMActivationFn = "relu",
      LSTMRecurrentActivationFn ="sigmoid" )
    # Previsao não interativa
    previ = predict(lstm,
                    horizon = p, # Previsão para todos os períodos da base de teste
                    xreg = X[1:t,],           # Covariáveis usadas no treino
                    ts = Y[1:t],             # Série temporal usada no treino
                    xreg.new = X[(t+1):(t+p),])         # Covariáveis da base de teste
    # Tabela para os grafico das previsões
    dados_prev_n_int = data.frame(data = base$DT_INTER[(t+1):(t+p)],previsao = previ,Internacao = Y[(t+1):(t+p)])
    # Previsão interativa
    pred_int = NULL
    for(i in (t+1):(t+p)){
      pred_int = c(pred_int,
                   predict(lstm, 
                           horizon = 1, 
                           xreg = X[1:(i-1),], 
                           ts = Y[1:(i-1)],
                           xreg.new = X[i, , drop = F]))
    }
    
    grafico = dados_prev_n_int |> ggplot(mapping = aes(x = data )) +
      geom_line(mapping = aes(y = previsao),color = "red") +
      geom_line(mapping = aes(y = Internacao),color = "black") + 
      geom_line(mapping = aes(y = pred_int),color = "blue")
    cat("lag = ",l," Neuro = ",n)
    grafico
  }

}


