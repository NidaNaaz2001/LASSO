 dev.off()
library(forecastML)
library(dplyr)
library(DT)

library(ggplot2)
library(glmnet)
library(randomForest)


data_seatbelts <- read.csv(file.choose(),header = TRUE ,sep =",")
data <- data_seatbelts
date_frequency <- "2 month"  # Time step frequency.


# The date indices, which don't come with the stock dataset, should not be included in the modeling data.frame.
dates <- seq(as.Date("1969-01-01"), as.Date("1984-12-01"), by = date_frequency)  #5805
#?round
#data$PetrolPrice <- round(data$PetrolPrice, 3)

data <- data[, c("index", "Voltage", "Charge_Capacity", "Discharge_Capacity")]
DT::datatable(head(data, 5))

#####Train-Test split
data_train <- data[1:(nrow(data) - 1500), ]
data_test <- data[(nrow(data) - 1500 + 1):nrow(data), ]

###Plot
p <- ggplot(data, aes(x = index, y = Voltage))
p <- p + geom_line()
p <- p + geom_vline(xintercept = dates[nrow(data_train)], color = "red", size = 1.1)
p <- p + theme_bw() + xlab("Dataset index")
p
###Data Preperation
#We’ll create a list of datasets for model training, one for each forecast horizon.
outcome_col <- 2  # The column index of our DriversKilled outcome.

horizons <- c(1)  # 4 models that forecast 1, 1:3, 1:6, and 1:12 time steps ahead.

# A lookback across select time steps in the past. Feature lags 1 through 9, for instance, will be
# silently dropped from the 12-step-ahead model.
lookback <- c(1:6, 9, 12, 15)

# A non-lagged feature that changes through time whose value we either know (e.g., month) or whose
# value we would like to forecast.
dynamic_features <- "Charge_Capacity"

data_list <- forecastML::create_lagged_df(data_train,
                                          outcome_col = outcome_col,
                                          type = "train",
                                          horizons = horizons,
                                          lookback = lookback,
                                          #date = dates[1:nrow(data_train)],
                                          frequency = date_frequency,
                                          dynamic_features = dynamic_features
)
# Tabulating the modelled dataset occurding to horizons

DT::datatable(head(data_list$horizon_6, 10), options = list(scrollX = TRUE))


# plot(data_list)

## Create windows
# ?create_windows
# windows <- forecastML::create_windows(lagged_df = data_list, window_length = 12, skip = 48,
#                                       window_start = NULL, window_stop = NULL,
#                                       include_partial_window = TRUE)
# windows
# skip = 48 which is an integer that indicates the no. of rows to skipped for later testing the model
#include_partial_window =  Boolean. If TRUE, keep validation datasets that are shorter than window_length.

## Plotting the windows in the trianing dataset
# plot(windows, data_list, show_labels = TRUE)
#
# ## Model trainig
# ## Comparing LASSO model with Random forest model
## 1)LASSO model
model_function <- function(data) {

  # The 'law' feature is constant during some of our outer-loop validation datasets so we'll
  # simply drop it so that glmnet converges.
  constant_features <- which(unlist(lapply(data[, -1], function(x) {!(length(unique(x)) > 1)})))

  if (length(constant_features) > 1) {
    data <- data[, -c(constant_features + 1)]  # +1 because we're skipping over the outcome column.
  }

  x <- data[, -(1), drop = FALSE]
  y <- data[, 1, drop = FALSE]
  x <- as.matrix(x, ncol = ncol(x))
  y <- as.matrix(y, ncol = ncol(y))

  model <- glmnet::cv.glmnet(x, y, nfolds = 3)
  return(list("model" = model, "constant_features" = constant_features))
}


# ## User defined prediction function
# Example 1 - LASSO.

prediction_function <- function(model, data_features) {

  if (length(model$constant_features) > 1) {  # 'model' was passed as a list.
    data_features <- data_features[, -c(model$constant_features )]
  }

  x <- as.matrix(data_features, ncol = ncol(data_features))

  data_pred <- data.frame("y_pred" = predict(model$model, x, s = "lambda.min"))
  return(data_pred)
}



######## As LASSO model is more accurate as it produces less error we consider LASSO model to retrain it again
# retrainig the LASSO model
data_list <- forecastML::create_lagged_df(data_train,
                                          outcome_col = outcome_col,
                                          type = "train",
                                          horizons = horizons,
                                          lookback = lookback,
                                          #date = dates[1:nrow(data_train)],
                                          frequency = date_frequency,
                                          dynamic_features = dynamic_features
)
# creating a window of length 0 and plotting it
windows <- forecastML::create_windows(data_list, window_length = 0)

plot(windows, data_list, show_labels = TRUE,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+ theme(
       # Hide panel borders and remove grid lines
       panel.grid.major = element_blank(),
       panel.grid.minor = element_blank(),text = element_text(size =15))


### Model output or prediction/Forecasting
model_results <- forecastML::train_model(data_list, windows,  model_name = "LASSO", model_function)

data_results <- predict(model_results, prediction_function = list(prediction_function), data = data_list)

DT::datatable(head(data_results, 10), options = list(scrollX = TRUE))

## ploting the forecast results

plot(data_results, type = "prediction",  horizons = c(1) ,
     panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+ theme(
  # Hide panel borders and remove grid lines
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),text = element_text(size =15)
)


## trainig error for 4 horizons
data_error <- forecastML::return_error(data_results)

data_error$error_global[, -1] <- lapply(data_error$error_global[, -1], round, 1)

DT::datatable(head(data_error$error_global), options = list(scrollX = TRUE))



# #############3Forecast with one model with horizon
# data_forecast_list <- forecastML::create_lagged_df(data_train,
#                                                    outcome_col = outcome_col,
#                                                    type = "forecast",
#                                                    horizons = horizons,
#                                                    lookback = lookback,
#                                                    #date = dates[1:nrow(data_train)],
#                                                    frequency = date_frequency,
#                                                    dynamic_features = dynamic_features
# )
#
# for (i in seq_along(data_forecast_list)) {
#   data_forecast_list[[i]]$Charge_Capacity <- 1
# }
#
# data_forecast <- predict(model_results, prediction_function = list(prediction_function), data = data_forecast_list)

# plot(data_forecast,
#      data_actual = data[-(67:1901), ],
#      actual_indices = dates[-(67:843)],
#      horizons = c(1, 6, 12))
# data_actual
###Forecast Error
# data_error <- forecastML::return_error(data_forecast, data_test = data_test,test_indices = dates[(nrow(data_train) + 1):nrow(data)])

plot(data_error, type = "horizon", facet = ~ horizon ,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+ theme(
       # Hide panel borders and remove grid lines
       panel.grid.major = element_blank(),
       panel.grid.minor = element_blank(),text = element_text(size =15))

