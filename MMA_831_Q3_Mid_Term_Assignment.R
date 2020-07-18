#Load Pacman and required libraries
if("pacman" %in% rownames(installed.packages())==FALSE){install.packages("pacman")}

pacman::p_load("tensorflow","keras","xgboost","dplyr","caret",
               "ROCR","lift","glmnet","MASS","e1071"
               ,"mice","partykit","rpart","randomForest","dplyr"   
               ,"lubridate","ROSE","smotefamily","DMwR","caretEnsemble"
               ,"MLmetrics")

#Load DataSet
data<-read.csv("C://Users//anuj//Documents//Anuj//MMA//Marketing Analytics//Mid Term Assignment//eureka_data_final_2019-01-01_2019-03-01.csv")

#Count of missing values per column
lapply(data, function(x) sum(is.na(x)))

str(data)

as.data.frame(colnames(data))

#Removing Region, Date, Client ID

data<-data[,-c(6,12,30)]

#Data Pre-processing

data$converted_in_7days<-ifelse(data$converted_in_7days>1,1,data$converted_in_7days)
data$converted_in_7days<-as.factor(data$converted_in_7days)
data$visited_air_purifier_page<-as.factor(data$visited_air_purifier_page)
data$visited_checkout_page<-as.factor(data$visited_checkout_page)
data$visited_contactus<-as.factor(data$visited_contactus)
data$visited_customer_service_amc_login<-as.factor(data$visited_customer_service_amc_login)
data$visited_demo_page<-as.factor(data$visited_demo_page)
data$visited_offer_page<-as.factor(data$visited_offer_page)
data$visited_security_solutions_page<-as.factor(data$visited_security_solutions_page)
data$visited_storelocator<-as.factor(data$visited_storelocator)
data$visited_successbookdemo<-as.factor(data$visited_successbookdemo)
data$visited_vacuum_cleaner_page<-as.factor(data$visited_vacuum_cleaner_page)
data$visited_water_purifier_page<-as.factor(data$visited_water_purifier_page)
data$visited_customer_service_request_login<-as.factor(data$visited_customer_service_request_login)
data$newUser<-as.factor(data$newUser)
data$fired_DemoReqPg_CallClicks_evt<-as.factor(data$fired_DemoReqPg_CallClicks_evt)
data$fired_help_me_buy_evt<-as.factor(data$fired_help_me_buy_evt)
data$fired_phone_clicks_evt<-as.factor(data$fired_phone_clicks_evt)
data$goal4Completions<-as.factor(data$goal4Completions)
data$paid<-as.factor(data$paid)


#Parsing Source Medium Feature
data$sourceMedium<-sub(".*/", "", data$sourceMedium)
trimws(data$sourceMedium,which = "left")

data$sourceMedium<-as.character(data$sourceMedium)

data$sourceMedium<-ifelse(data$sourceMedium==" Social"," social",data$sourceMedium)
data$sourceMedium<-ifelse(data$sourceMedium==" (none)","None",data$sourceMedium)
data$sourceMedium<-ifelse(data$sourceMedium==" (not set)","None",data$sourceMedium)
data$sourceMedium<-trimws(data$sourceMedium)
data$sourceMedium<-as.factor(data$sourceMedium)


levels(data$sourceMedium)




# Create a custom function to fix missing values ("NAs") and preserve the NA info as surrogate variables
fixNAs<-function(data_frame){
  # Define reactions to NAs
  integer_reac<-0
  factor_reac<-"FIXED_NA"
  character_reac<-"FIXED_NA"
  date_reac<-as.Date("1900-01-01")
  # Loop through columns in the data frame and depending on which class the variable is, apply the defined reaction and create a surrogate
  
  for (i in 1 : ncol(data_frame)){
    if (class(data_frame[,i]) %in% c("numeric","integer")) {
      if (any(is.na(data_frame[,i]))){
        data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
          as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
        data_frame[is.na(data_frame[,i]),i]<-integer_reac
      }
    } else
      if (class(data_frame[,i]) %in% c("factor")) {
        if (any(is.na(data_frame[,i]))){
          data_frame[,i]<-as.character(data_frame[,i])
          data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
            as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
          data_frame[is.na(data_frame[,i]),i]<-factor_reac
          data_frame[,i]<-as.factor(data_frame[,i])
          
        } 
      } else {
        if (class(data_frame[,i]) %in% c("character")) {
          if (any(is.na(data_frame[,i]))){
            data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
              as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
            data_frame[is.na(data_frame[,i]),i]<-character_reac
          }  
        } else {
          if (class(data_frame[,i]) %in% c("Date")) {
            if (any(is.na(data_frame[,i]))){
              data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
                as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
              data_frame[is.na(data_frame[,i]),i]<-date_reac
            }
          }  
        }       
      }
  } 
  return(data_frame) 
}


data1<-fixNAs(data)

#Smote Balancing
data_balance<-SMOTE(converted_in_7days~.,data=data1,perc.over=200,perc.under=200)

table(data_balance$converted_in_7days)

str(data_balance)

#Data Split

set.seed(77850) #set a random number generation seed to ensure that the split is the same everytime
inTrain <- createDataPartition(y = data_balance$converted_in_7days,
                               p = 0.8, list = FALSE)
training <- data_balance[ inTrain,]
testing <- data_balance[ -inTrain,]

data_matrix<-data.matrix(dplyr::select(data_balance,-converted_in_7days))

x_train <- data_matrix[ inTrain,]
x_test <- data_matrix[ -inTrain,]

y_train <-training$converted_in_7days
y_test <-testing$converted_in_7days

#Ensemble Model with GLM method

ctrl <- trainControl(
  method="boot",
  number=10,
  savePredictions="final",
  classProbs=TRUE,
  summaryFunction=twoClassSummary,
  allowParallel = TRUE
)


model_list <- caretList(
  x=x_train,y=make.names(y_train),  
  trControl=ctrl,
  methodList=c("xgbTree", "rf")
)


stack_fit <- caretStack(
  model_list,
  method="glm",
  metric="ROC",
  trControl=trainControl(
    method="boot",
    number=10,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)
summary(stack_fit)
stack_pred <-predict(stack_fit, x_test)
stack_pred<-as.factor(ifelse(stack_pred=="X0",0,1))

stack_fit$ens_model

#Confusion Matrix
caret::confusionMatrix(data=stack_pred, reference=y_test, positive="1", dnn=c("Predicted", "Actual"))

#AUC
AUC(stack_pred,y_test)

