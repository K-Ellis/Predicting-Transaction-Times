library(ggplot2)

RMSE = c(129.4, 88.1, 88.3, 41.0, 23.7)
# RMSE_std = c(0,2.33, 2.31,0.91,1.13 )
# R2 = c()
MeanAE = c(90.8,  54.9,54.9 ,26.4,7.9)
# MedianAE = c()
Algorithms = c("B", "LR", "EN", "GBR", "RFR")
df = data.frame(Algorithms,RMSE, MeanAE)
df$Algorithms = factor(df$Algorithms, as.character(df$Algorithms))
# mention 10 fold CV is where the errors came from somewhere

# RMSE plot
ggplot(data=df, aes(x=Algorithms, y=RMSE, fill=RMSE))+
  geom_bar(position=position_dodge(),stat="identity")+ #, color="blue")+
  geom_text(aes(label=RMSE), vjust=1.5, color="white",position = position_dodge(0.9), size=5)+
  ggtitle("RMSE for each algorithm") +
  theme(plot.title = element_text(hjust = 0.5))+
  guides(fill=FALSE)+
  theme_minimal() 

library(reshape2)
errors = melt(df, id="Algorithms")
errors$ErrorName = errors$variable
errors$ErrorValue = errors$value
errors
# library(dplyr)
# bar <- group_by(errors, variable, gender)%.%summarise(mean=mean(value))

# RMSE MeanAE plot
ggplot(data=errors, aes(Algorithms, ErrorValue, fill=ErrorName))+
  geom_bar(position="dodge",stat="identity")+
  geom_text(aes(label=ErrorValue), vjust=.9, color="black",position = position_dodge(0.9), size=5)+
  # ggtitle("RMSE and MAE for each algorithm") +
  # theme(plot.title = element_text(hjust = 0.5)) +
  theme_minimal()+
  labs(y="Error Value", title="RMSE and MAE for each algorithm")

PredictionsWithin96hours = c(74.9, 86.1, 85.7, 96.6, 99.0)
PredictionsWithin96hourspct = c("74.9%", "86.1%", "85.7%", "96.6%", "99%")
df$PredictionsWithin96hours = PredictionsWithin96hours
df$PredictionsWithin96hourspct = PredictionsWithin96hourspct

ggplot(data=df, aes(x=Algorithms, y=PredictionsWithin96hours, fill=PredictionsWithin96hours))+
  geom_bar(position=position_dodge(),stat="identity")+ #, color="blue")+
  geom_text(aes(label=PredictionsWithin96hourspct), vjust=1.5, color="white",position = position_dodge(0.9), size=5)+
  ggtitle("Predictions Within 96 Hours for each Algorithm") +
  theme(plot.title = element_text(hjust = 0.5))+
  guides(fill=FALSE)+
  coord_cartesian(ylim = c(70, 100)) + 
  scale_y_continuous(labels = function(x){ paste0(x, "%") })+
  theme_minimal()


pct_hours = c(1,4,8,16,24,48,72,96)
B = c(0.0091,0.0341,0.0562,0.0806,0.1412,0.2782,0.4348,0.7493)
RFR = c(0.3541,0.6698,0.7946,0.8867,0.9256,0.9668,0.9822,0.99)
EN = c(0.0264, .1010, .1928, .3109, .4113, .6062, .7487, .8574)
LR = c(2.52,9.98,19.09,30.85,40.78,60.45,75.33,86.11)
LR = LR/100
GBR = c(3.87,15.41,28.71,49.09,62.27,84.92,92.94,96.61)
GBR = GBR/100
pctdf = data.frame(pct_hours, B, RFR, EN, LR, GBR)

pct_long = melt(pctdf, id="pct_hours")
pct_long$Algorithm = pct_long$variable
ggplot(data=pct_long, aes(x=pct_hours, y=value, colour=Algorithm)) +
  geom_line(size=1)+  
  geom_point(size=1.5)+
  theme_minimal()

plot.table(table(Algorithms, RMSE))
