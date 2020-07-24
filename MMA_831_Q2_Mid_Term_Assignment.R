#Load Pacman and required libraries
if("pacman" %in% rownames(installed.packages())==FALSE){install.packages("pacman")}

pacman::p_load("tensorflow","keras","xgboost","dplyr","caret",
               "ROCR","lift","glmnet","MASS","e1071"
               ,"mice","partykit","rpart","randomForest","dplyr"   
               ,"lubridate","ROSE","smotefamily","DMwR","beepr","MLmetrics",
               "caretEnsemble","mlbench","gbm","gdata")

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

#Principle Component Analysis(This is not a mandatory step)
data1$fired_DemoReqPg_CallClicks_evt<-as.numeric(data1$fired_DemoReqPg_CallClicks_evt)
data1$visited_demo_page<-as.numeric(data1$visited_demo_page)
data1$visited_successbookdemo<-as.numeric(data1$visited_successbookdemo)
data1$converted_in_7days<-as.numeric(data1$converted_in_7days)

str(data1)

data1.pca<-prcomp(data1[,c(1,2,3,4,5,6,7,9,10,11,13,14,18,19,21,
                           22,23,25,26,27,28,29,30,31,32,34,35,
                           36,38,40,42,44,46,47,48,50,52,54,55,57,59,60
                           )],center =TRUE, scale. = TRUE)

summary(data1.pca)
str(data1.pca)
data1.pca$rotation

#Data Split

set.seed(77850) #set a random number generation seed to ensure that the split is the same everytime
inTrain <- createDataPartition(y = data1$converted_in_7days,
                               p = 0.8, list = FALSE)
training <- data1[ inTrain,]
testing <- data1[ -inTrain,]

table(training$converted_in_7days)

#Smote Balancing
data_balance<-SMOTE(converted_in_7days~.,data=training,perc.over=200)

table(data_balance$converted_in_7days)

str(data_balance)


#Hyperparameter Tuning
control<-trainControl(method="repeatedcv",number=10,repeats=1,search="random")
start_time_rf<-Sys.time()
for(i in c(25,50,100)){
  rf_random<-train(converted_in_7days~.,
                   data=data_balance,ntree=i,method="rf",metric="Accuracy",tuneLength=3,trControl=control)
  print(i)
  print(rf_random)
  plot(rf_random)
}
end_time_rf<-Sys.time()

#-----Random Forest(Enter the ntree and mtry based on the results of Cross Validation)
model_forest <- randomForest(converted_in_7days~., data=data_balance, 
                             type="classification",
                             importance=TRUE,
                             ntree = 100,                                    
                             mtry = 10,             
                             nodesize = 10,         
                             maxnodes = 10,         
                             cutoff = c(0.5, 0.5)   
) 

beep(8)
plot(model_forest)  
varImpPlot(model_forest) 

###Finding predicitons: probabilities and classification
forest_probabilities<-predict(model_forest,newdata=testing,type="prob") 
forest_classification<-rep("1",141865)
forest_classification[forest_probabilities[,2]<0.2]="0" 
forest_classification<-as.factor(forest_classification)

confusionMatrix(forest_classification,testing$converted_in_7days, positive="1") 



####ROC Curve
forest_ROC_prediction <- prediction(forest_probabilities[,2], testing$converted_in_7days) 
forest_ROC <- performance(forest_ROC_prediction,"tpr","fpr") 
plot(forest_ROC) 

####AUC (area under curve)
AUC.tmp <- performance(forest_ROC_prediction,"auc") 
forest_AUC <- as.numeric(AUC.tmp@y.values) 
forest_AUC 

#### Lift chart
plotLift(forest_probabilities[,2],  testing$converted_in_7days, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

Lift_forest <- performance(forest_ROC_prediction,"lift","rpp")
plot(Lift_forest)


set.seed(77850) #set a random number generation seed to ensure that the split is the same everytime
inTrain1 <- createDataPartition(y = data_balance$converted_in_7days,
                               p = 0.8, list = FALSE)
training1 <- data_balance[ inTrain1,]
testing1 <- data_balance[ -inTrain1,]


data_matrix<-data.matrix(dplyr::select(data_balance,-converted_in_7days))
data_matrix1<-data.matrix(dplyr::select(data1,-converted_in_7days))


x_train <- data_matrix[ inTrain1,]
x_test <- data_matrix1[ -inTrain,]

y_train <-training1$converted_in_7days
y_test <-testing$converted_in_7days

#xgBoost Hyper-parameter tuning
control<-trainControl(method="repeatedcv",
                      number=5,repeats=1,search="random")

start_time_xg<-Sys.time()

xg_Boost<-train(y=y_train,
                   x=x_train,method="xgbTree",metric="Accuracy",
                tuneLength=3,trControl=control)

end_time_xg<-Sys.time()
beep(8)

#Print summary of all Cross validations
print(xg_Boost)

#Print hyperparameters best tuned model
xg_Boost$bestTune

#Calculate Probability and build confusion matrix
XGboost_prediction<-predict(xg_Boost,newdata=x_test,type="prob") 
confusionMatrix(as.factor(ifelse(XGboost_prediction[,2]>0.4,1,0)),y_test,positive="1") 
####ROC Curve
XGboost_ROC_prediction <- prediction(XGboost_prediction[,2], y_test) 
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") 
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") 
XGboost_auc_testing <- as.numeric(auc.tmp@y.values)
XGboost_auc_testing 

#### Lift chart
plotLift(XGboost_prediction, y_test, cumulative = TRUE, n.buckets = 10)

Lift_XGboost <- performance(XGboost_ROC_prediction,"lift","rpp")
plot(Lift_XGboost)    
