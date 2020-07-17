#Load Pacman and required libraries
if("pacman" %in% rownames(installed.packages())==FALSE){install.packages("pacman")}

pacman::p_load("tensorflow","keras","xgboost","dplyr","caret",
               "ROCR","lift","glmnet","MASS","e1071"
               ,"mice","partykit","rpart","randomForest","dplyr"   
               ,"lubridate","ROSE","smotefamily","DMwR")

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

#Principle Component Analysis(This is not a mandatory step)
str(data1)


data1$fired_DemoReqPg_CallClicks_evt<-as.numeric(data1$fired_DemoReqPg_CallClicks_evt)
data1$visited_demo_page<-as.numeric(data1$visited_demo_page)
data1$demoSales<-as.numeric(data1$demoSales)
data1$visited_successbookdemo<-as.numeric(data1$visited_successbookdemo)
data1$converted_in_7days<-as.numeric(data1$converted_in_7days)


data1.pca<-prcomp(data1[,c(1,2,3,4,5,7,8,10,11,13,15,16,20,21,23,24,25,
                           27,28,29,31,32,33,34,35,37,38,39,41,43,45,
                           47,49,50,51,53,55,57,58,60,62,63,65)],center =TRUE, scale. = TRUE)

summary(data1.pca)
str(data1.pca)
data1.pca$rotation

#Data Split

set.seed(77850) #set a random number generation seed to ensure that the split is the same everytime
inTrain <- createDataPartition(y = data_balance$converted_in_7days,
                               p = 0.8, list = FALSE)
training <- data_balance[ inTrain,]
testing <- data_balance[ -inTrain,]


#Hyperparameter Tuning
control<-trainControl(method="repeatedcv",number=10,repeats=1,search="random")
start_time_rf<-Sys.time()
for(i in c(50,100,150)){
  rf_random<-train(converted_in_7days~.,
                   data=training,ntree=i,method="rf",metric="Accuracy",tuneLength=3,trControl=control)
  print(i)
  print(rf_random)
  plot(rf_random)
}
end_time_rf<-Sys.time()

#-----Random Forest(Enter the ntree and mtry based on the results of Cross Validation)
model_forest <- randomForest(converted_in_7days~., data=training, 
                             type="classification",
                             importance=TRUE,
                             ntree = 25,           # hyperparameter: number of trees in the forest
                             mtry = 10,             # hyperparameter: number of random columns to grow each tree
                             nodesize = 10,         # hyperparameter: min number of datapoints on the leaf of each tree
                             maxnodes = 10,         # hyperparameter: maximum number of leafs of a tree
                             cutoff = c(0.5, 0.5)   # hyperparameter: how the voting works; (0.5, 0.5) means majority vote
) 

plot(model_forest)  # plots error as a function of number of trees in the forest; use print(model_forest) to print the values on the plot

varImpPlot(model_forest) # plots variable importances; use importance(model_forest) to print the values


###Finding predicitons: probabilities and classification
forest_probabilities<-predict(model_forest,newdata=testing,type="prob") #Predict probabilities -- an array with 2 columns: for not retained (class 0) and for retained (class 1)
forest_classification<-rep("1",3962)
forest_classification[forest_probabilities[,2]<0.4]="0" #Predict classification using 0.5 threshold. Why 0.5 and not 0.6073? Use the same as in cutoff above
forest_classification<-as.factor(forest_classification)

confusionMatrix(forest_classification,testing$converted_in_7days, positive="1") #Display confusion matrix. Note, confusion matrix actually displays a better accuracy with threshold of 50%

#There is also a "shortcut" forest_prediction<-predict(model_forest,newdata=testing, type="response") 
#But it by default uses threshold of 50%: actually works better (more accuracy) on this data


####ROC Curve
forest_ROC_prediction <- prediction(forest_probabilities[,2], testing$converted_in_7days) #Calculate errors
forest_ROC <- performance(forest_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(forest_ROC) #Plot ROC curve

####AUC (area under curve)
AUC.tmp <- performance(forest_ROC_prediction,"auc") #Create AUC data
forest_AUC <- as.numeric(AUC.tmp@y.values) #Calculate AUC
forest_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(forest_probabilities[,2],  testing$converted_in_7days, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_forest <- performance(forest_ROC_prediction,"lift","rpp")
plot(Lift_forest)


#xgBoost
#data_matrix <- model.matrix(converted_in_7days~ .-region-client_id-date-sourceMedium, data = data1_balance)[,-1]

data_matrix<-data.matrix(dplyr::select(data_balance,-converted_in_7days))

x_train <- data_matrix[ inTrain,]
x_test <- data_matrix[ -inTrain,]

y_train <-training$converted_in_7days
y_test <-testing$converted_in_7days

#xgBoost Hyper-parameter tuning
control<-trainControl(method="repeatedcv",
                      number=5,repeats=1,search="random")

start_time_xg<-Sys.time()

xg_Boost<-train(y=y_train,
                   x=x_train,method="xgbTree",metric="accuracy",
                tuneLength=3,trControl=control)

end_time_xg<-Sys.time()

#Print summary of all Cross validations
print(xg_Boost)

#Print hyperparameters best tuned model
xg_Boost$bestTune

#Calculate Probability and build confusion matrix
XGboost_prediction<-predict(xg_Boost,newdata=x_test,type="prob") #Predict classification (for confusion matrix)
confusionMatrix(as.factor(ifelse(XGboost_prediction[,2]>0.4,1,0)),y_test,positive="1") #Display confusion matrix

####ROC Curve
XGboost_ROC_prediction <- prediction(XGboost_prediction[,2], y_test) #Calculate errors
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") #Create AUC data
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
XGboost_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(XGboost_prediction, y_test, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_XGboost <- performance(XGboost_ROC_prediction,"lift","rpp")
plot(Lift_XGboost)



