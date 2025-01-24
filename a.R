# Pacotes
{
  library(haven)
  library(dplyr)
  library(naniar)
  library(ggplot2)
  library(caret)
  library(TSLSTMplus)
}

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

base = base |> select(DT_INTER,temperatura_c_media,pm25_ugm3_medio,casos_resp) 

Y = base$casos_resp
X = base |> 
  select(-c(DT_INTER,casos_resp))


N = nrow(base)
k = nrow(base |> filter(DT_INTER <= "2019-08-31")) #ultimo instante do treino

lstm <- ts.lstm(
  ts = Y[1:k],
  xreg = X[1:k,],
  tsLag = 1,
  xregLag = 1,
  LSTMUnits = c(20,10), 
  ScaleInput = "minmax",
  ScaleOutput = "minmax",BatchSize = 64,RandomState = 122054005)


#prevendo hoje para amanhã
t = k + 1
predict(lstm,
        horizon = 1,
        xreg = X[(1:(t-1)),],
        ts = Y[(1:(t-1))],
        xreg.new = X[t,])

#média movel de hoje para amanha
#considerando janela igual a lag
mean(Y[(t-lag):(t-1)])
Y[t]

#prevendo de um dia ptara o seguinte
#para os útlimos 31 dias
lag1 = 2
pred = Y[1:lag1]
for(t in (lag1+1):N){
  pred = c(pred,
           predict(lstm, 
                   horizon = 1, 
                   xreg = X[(1:(t-1)),], 
                   ts = Y[(1:(t-1))],
                   xreg.new = X[t,]))
  
}


##########################################
# Modelo 2

lstm2 <- ts.lstm(
  ts = Y[1:k],
  xreg = X[1:k,],
  tsLag = 2,
  xregLag = 2,
  LSTMUnits = c(20,10), 
  ScaleInput = "minmax",
  ScaleOutput = "minmax",
  BatchSize = 64,RandomState = 122054005)

lag2 = 2
pred2 = Y[1:lag2]
for(t in (lag2+1):N){
  pred2 = c(pred2,
           predict(lstm2, 
                   horizon = 1, 
                   xreg = X[(1:(t-1)),], 
                   ts = Y[(1:(t-1))],
                   xreg.new = X[t,]))
  
}

##########################################
# Modelo 3

lstm3 <- ts.lstm(
  ts = Y[1:k],
  xreg = X[1:k,],
  tsLag = 3,
  xregLag = 3,
  LSTMUnits = c(20,10), 
  ScaleInput = "minmax",
  ScaleOutput = "minmax",
  BatchSize = 64,RandomState = 122054005)

lag3 = 3
pred3 = Y[1:lag3]
for(t in (lag3+1):N){
  pred3 = c(pred3,
            predict(lstm3, 
                    horizon = 1, 
                    xreg = X[(1:(t-1)),], 
                    ts = Y[(1:(t-1))],
                    xreg.new = X[t,]))
  
}

##########################################
# Modelo 4

lstm4 <- ts.lstm(
  ts = Y[1:k],
  xreg = X[1:k,],
  tsLag = 7,
  xregLag = 7,
  LSTMUnits = c(20,10), 
  ScaleInput = "minmax",
  ScaleOutput = "minmax",
  BatchSize = 64,RandomState = 122054005)

lag4 = 7
pred4 = Y[1:lag4]
for(t in (lag4+1):N){
  pred4 = c(pred4,
            predict(lstm4, 
                    horizon = 1, 
                    xreg = X[(1:(t-1)),], 
                    ts = Y[(1:(t-1))],
                    xreg.new = X[t,]))
  
}



# Verificando no grafico
plot(x = base$DT_INTER[1:N],y = Y[1:N],type = "l",ylab = "N internação",xlab = "data")
points(x= base$DT_INTER[1:N],y=pred,col="red",type = "l",) # Mod 1
points(x=base$DT_INTER[1:N],y=pred2,col="blue",type = "l") # Mod 2
points(x=base$DT_INTER[1:N],y=pred3,col="orange",type = "l")
points(x=base$DT_INTER[1:N],y=pred4,col="green",type = "l")

postResample(pred,Y[1:N])
postResample(pred2,Y[1:N])
postResample(pred3,Y[1:N])
postResample(pred4,Y[1:N])
