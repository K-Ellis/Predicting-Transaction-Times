# install.packages("tree")
library(tree)
install.packages("rpart")
library(rpart)
set.seed(123456789)

# Import data
filename <- #"C:/Users/Office/Google Drive/1. UCD Business Analytics/11. Practicum/Microsoft Project/Eoin Folder/SampleData2.xlsx"
SampleData <- read_excel(filename)

str(SampleData)

# Pre-process data, manually done this time:
# Delete all columns where everything is the same eg. all ones
# Delete rows with NULL resolved date
# Deleted Outliers - eg. really tiny time to process
# Deleted all rows with a NULL in them

# Histogram of times - all data
hist(SampleData$Days, main = "Histogram of times to completion (all data)", xlab = "Days")
# Histogram of times - less than one day
hist(SampleData$Days[SampleData$Days<1], main = "Histogram of times to completion (<1 day)", xlab = "Days")
# Histogram of times - less than 5 hours
hist(SampleData$Days[SampleData$Days<0.2], main = "Histogram of times to completion (<5 hrs)", xlab = "Days")
# Histogram of times - less than 1.2 hours
hist(SampleData$Days[SampleData$Days<0.05], main = "Histogram of times to completion (<1 hr)", xlab = "Days")

# Regression Tree
# Split data
ind = sample(2,nrow(SampleData),replace=TRUE,prob=c(0.7,0.3))
trainData = SampleData[ind==1,]
testData = SampleData[ind==2,]
myFormula = Days ~ Queue + RevenuType + Priority + Program + CaseType + Reason + CountryProcessed
#tree.SampleData = tree(myFormula,trainData)
#tree.SampleData = tree(trainData$Days~.,trainData)
rpart.SampleData = rpart(myFormula,data=trainData,method="class")

summary(tree.SampleData)
tree.SampleData

plot(tree.SampleData)
text(tree.SampleData,all=TRUE,cex=0.5)
