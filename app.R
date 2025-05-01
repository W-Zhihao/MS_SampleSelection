# app.R

# Load required libraries
library(shiny)
library(caret)
library(e1071)        # for SVM
library(randomForest) # for Random Forest
library(nnet)         # for neural networks
library(mgcv)         # for GAM
library(httr)         # for API calls (if needed)
library(DT)           # for editable tables

# Increase max upload size (e.g., 10 MB limit)
options(shiny.maxRequestSize = 10 * 1024^2)

ui <- fluidPage(
  titlePanel("Margin Sampling Active Learning Strategy with Different Models for Selecting Informative Samples"),
  sidebarLayout(
    sidebarPanel(
      h4("1. Upload Data"),
      fileInput("train_file", "Training Data (CSV with x, y, label)", accept = ".csv"),
      tags$small("Ensure: 'x' and 'y' are coordinates (rounded to 3 decimal places), and 'label' uses 1 for landslide and 0 for non-landslide."),
      fileInput("unlab_file", "Unlabeled Data (CSV with x, y, features)", accept = ".csv"),
      tags$hr(),
      
      h4("2. Choose Model & Parameters"),
      selectInput("model_type", "Model:", choices = c("SVM", "Random Forest", "ANN", "GAM")),
      numericInput("cv_folds", "CV Folds:", value = 5, min = 2),
      numericInput("cv_repeats", "CV Repeats:", value = 1, min = 1),
      numericInput("tune_length", "Tuning Complexity (tuneLength):", value = 3, min = 1),
      actionButton("train_btn", "Train Model", class = "btn-primary"),
      tags$hr(),
      
      h4("3. Active Learning"),
      numericInput("num_uncertain", "Number of Uncertain Samples to query:", value = 5, min = 1),
      actionButton("get_uncertain", "Select Uncertain Samples"),
      tags$hr(),
      
      h4("4. Add New Labels"), # & Retrain
      tags$small("Option 1: Upload a CSV for the uncertain samples with corrected labels."),
      fileInput("edited_labels_file", "Upload Edited Uncertain Samples (CSV)", accept = ".csv"),
      br(),
      tags$small("Option 2 (Optional): Upload additional new training data (CSV)."),
      fileInput("additional_train_file", "Upload Additional New Training Data (CSV)", accept = ".csv"),
      actionButton("update_data", "Update Data", class = "btn-warning"),
      # actionButton("retrain_model", "Retrain Model", class = "btn-info"),
      tags$hr(),
      
      h4("Downloads"),
      # downloadButton("download_predictions", "Download All Predictions"),
      downloadButton("download_training", "Download training Samples")
    ),
    
    mainPanel(
      h3("Training Data Preview"),
      tableOutput("train_preview"),
      
      h3("Unlabeled Data Preview"),
      tableOutput("unlab_preview"),
      
      h3("Model Training Results"),
      verbatimTextOutput("train_results"),
      
      h3("Unlabeled Data Predictions"),
      tableOutput("predictions_table"),
      
      h3("Most Uncertain Samples (Editable)"),
      DTOutput("uncertain_table_edit"),
      
      # "Save Edited Uncertain Samples" button
      actionButton("save_edits", "Save Edited Uncertain Samples", class = "btn-secondary"),
      
      h3("Extra Selected Data Upload from CSV"),
      tableOutput("extra_data"),
      
      h3("Updated Training Data Preview"),
      tableOutput("updated_train_table"),
      
      textOutput("status_msg")
    )
  )
)

server <- function(input, output, session) {
  # Reactive values for data and model
  trainingData <- reactiveVal(NULL)
  unlabeledData <- reactiveVal(NULL)
  trainedModel <- reactiveVal(NULL)
  predictions <- reactiveVal(NULL)
  uncertainSamples <- reactiveVal(NULL)
  updatedTrainData <- reactiveVal(NULL)
  extradata <- reactiveVal(NULL)
  
  # 1. Upload Training Data
  observeEvent(input$train_file, {
    req(input$train_file)
    cat("Training file uploaded:", input$train_file$name, "\n")
    df <- tryCatch(read.csv(input$train_file$datapath, header = TRUE),
                   error = function(e) { NULL })
    if (is.null(df)) {
      showNotification("Error reading training CSV. Please check file format.", type = "error")
      return()
    }
    if (!all(c("x", "y", "label") %in% names(df))) {
      showNotification("Training data must have columns: x, y, label", type = "error")
      return()
    }
    # Round x and y columns to 3 decimal places
    df$x <- round(as.numeric(df$x), 3)
    df$y <- round(as.numeric(df$y), 3)
    df$label <- as.factor(df$label)
    if (!all(c("0", "1") %in% levels(df$label))) {
      showNotification("Label column must use '1' for landslide and '0' for non-landslide.", type = "error")
      return()
    }
    # Rename factor levels
    df$label <- factor(df$label, levels = c("0", "1"), labels = c("nonlandslide", "landslide"))
    trainingData(df)
    # updatedTrainData(df)
    showNotification("Training data uploaded successfully.", type = "message")
  })
  
  # 1. Upload Unlabeled Data
  observeEvent(input$unlab_file, {
    req(input$unlab_file)
    cat("Unlabeled file uploaded:", input$unlab_file$name, "\n")
    df <- tryCatch(read.csv(input$unlab_file$datapath, header = TRUE),
                   error = function(e) { NULL })
    if (is.null(df)) {
      showNotification("Error reading unlabeled CSV. Please check file format.", type = "error")
      return()
    }
    if (!all(c("x", "y") %in% names(df))) {
      showNotification("Unlabeled data must have at least columns: x, y", type = "error")
      return()
    }
    # Round x and y columns to 3 decimals
    df$x <- round(as.numeric(df$x), 3)
    df$y <- round(as.numeric(df$y), 3)
    if ("label" %in% names(df)) {
      df$label <- NULL
      showNotification("Note: 'label' column in unlabeled data was ignored.", type = "warning")
    }
    unlabeledData(unique(df))
    showNotification("Unlabeled data uploaded successfully.", type = "message")
  })
  
  # Show training data preview
  output$train_preview <- renderPrint({
    req(trainingData())
    str(trainingData())#, 10
  })
  
  # Show unlabeled data preview
  output$unlab_preview <- renderPrint({
    req(unlabeledData())
    str(unlabeledData())#, 10
  })
  
  # 2. Train Model using caret
  observeEvent(input$train_btn, {
    req(trainingData())
    model_method <- switch(
      input$model_type,
      "SVM" = "svmRadial",
      "Random Forest" = "rf",
      "ANN" = "nnet",
      "GAM" = "gam"
    )
    cv_ctrl <- trainControl(
      method = "cv", number = input$cv_folds,
      repeats = if (input$cv_repeats > 1) input$cv_repeats else 1,
      classProbs = TRUE
    )
    tryCatch({
      withProgress(message = paste("Training", input$model_type, "model..."), {
        model <- train(label ~ ., data = trainingData()[,!(names(trainingData()) %in% c('x','y'))],
                       method = model_method,
                       trControl = cv_ctrl,
                       tuneLength = input$tune_length)
        trainedModel(model)
      })
      output$train_results <- renderPrint({ trainedModel() })
      best <- trainedModel()$bestTune
      bestText <- paste(names(best), "=", best, collapse = ", ")
      showNotification(paste("Model trained. Best parameters:", bestText), type = "message")
    }, error = function(e) {
      showNotification(paste("Error during training:", e$message), type = "error")
    })
    
    if (!is.null(unlabeledData()) && !is.null(trainedModel())) {
      probs <- predict(trainedModel(), newdata = unlabeledData(), type = "prob")
      preds <- predict(trainedModel(), newdata = unlabeledData(), type = "raw")
      if (all(c("1", "0") %in% names(probs))) {
        names(probs)[names(probs) == "1"] <- "landslide"
        names(probs)[names(probs) == "0"] <- "nonlandslide"
      }
      result_df <- cbind(unlabeledData(), PredictedLabel = preds, probs)
      result_df <- unique(result_df)
      predictions(result_df)
      output$predictions_table <- renderTable({ head(result_df, 10) }) #renderTable({result_df}) 
      showNotification("Predictions on unlabeled data generated.", type = "message")
    }
  })
  
  # 3. Active Learning: Select Most Uncertain Samples using Margin Sampling
  observeEvent(input$get_uncertain, {
    req(predictions())
    pred_df <- predictions()
    # Check if probability columns exist with explicit names; otherwise, auto-detect
    if (all(c("nonlandslide", "landslide") %in% names(pred_df))) {
      prob_cols <- c("nonlandslide", "landslide")
    } else {
      candidate_cols <- setdiff(names(pred_df), c("x", "y", "PredictedLabel"))
      prob_cols <- candidate_cols[sapply(pred_df[, candidate_cols, drop = FALSE], is.numeric)]
      prob_cols <- prob_cols[sapply(prob_cols, function(col) {
        med_val <- median(as.numeric(pred_df[[col]]), na.rm = TRUE)
        med_val >= 0 && med_val <= 1
      })]
    }
    if (length(prob_cols) < 2) {
      showNotification("Probability predictions not available to compute uncertainty.", type = "error")
      return()
    }
    prob_matrix <- as.matrix(sapply(pred_df[, prob_cols, drop = FALSE], as.numeric))
    if (any(is.na(prob_matrix))) {
      showNotification("Some prediction probabilities contain NA values.", type = "error")
      return()
    }
    top2diff <- apply(prob_matrix, 1, function(p) {
      sorted_probs <- sort(p, decreasing = TRUE)
      sorted_probs[1] - sorted_probs[2]
    })
    # Check that requested number does not exceed available samples
    num_requested <- input$num_uncertain
    if (num_requested > nrow(pred_df)) {
      num_requested <- nrow(pred_df)
      showNotification("Requested number exceeds available unlabeled samples. Adjusting to available count.", type = "warning")
    }
    idx <- order(top2diff)[1:num_requested]
    uncertain_df <- pred_df[idx, ]
    uncertainSamples(uncertain_df)
    output$uncertain_table_edit <- DT::renderDT({
      DT::datatable(uncertain_df, editable = TRUE)
    })
    showNotification(paste("Selected top", num_requested, "uncertain samples for labeling."), type = "message")
  })
  
  # Listen for edits in the DT table
  observeEvent(input$uncertain_table_edit_cell_edit, {
    req(uncertainSamples())
    info <- input$uncertain_table_edit_cell_edit
    df <- uncertainSamples()
    df[info$row, info$col] <- info$value
    uncertainSamples(df)
    DT::replaceData(DT::dataTableProxy("uncertain_table_edit"), df, resetPaging = FALSE)
  })
  
  # 4. Update Training Data: Update new labels and remove them from unlabeled data
  observeEvent(input$update_data, {
    req(trainingData())
    new_labels <- NULL
    # Option 1: Use edited uncertain samples from the GUI if available
    if (!is.null(input$edited_labels_file)) {
      new_labels <- tryCatch(
        read.csv(input$edited_labels_file$datapath, header = TRUE),
        error = function(e) { NULL }
      )
      
      if (is.null(new_labels) || !all(c("x", "y", "label") %in% names(trainingData()))) {
        showNotification("Uploaded Edited Uncertain Samples CSV is invalid.", type = "error")
        return()
      }
      
      selected <- uncertainSamples()
      # Round x and y in unlabeled data to 3 decimals for matching
      selected$x <- round(as.numeric(selected$x), 3)
      selected$y <- round(as.numeric(selected$y), 3)
      new_labels$x <- round(as.numeric(new_labels$x), 3)
      new_labels$y <- round(as.numeric(new_labels$y), 3)
      csv_ids <- paste(new_labels$x, new_labels$y, sep = "_")
      selected_ids <- paste(selected$x, selected$y, sep = "_")
      
      if (!all(csv_ids %in% selected_ids)) {
        showNotification("Uploaded Edited Uncertain Samples CSV is not selected one.", type = "error")
        return()
      }
      
      extradata(new_labels)
      
      # Display extra data preview (first 10 rows)
      output$extra_data <- renderPrint({
        req(extradata())
        str(extradata())#, 10
      })
    } else  if (!is.null(uncertainSamples())) {
     new_labels <- uncertainSamples()
     new_labels$label <- new_labels$PredictedLabel
     new_labels$PredictedLabel <- NULL
   }  else {
      showNotification("Please edit uncertain sample labels via the table or upload a CSV.", type = "error")
      return()
    }
    # Round x and y columns for new labels
    new_labels$x <- round(as.numeric(new_labels$x), 3)
    new_labels$y <- round(as.numeric(new_labels$y), 3)
    new_labels$label <- as.factor(new_labels$label)
    
    # Option 2: Additional new training data (optional)
    additional_df <- NULL
    if (!is.null(input$additional_train_file)) {
      additional_df <- tryCatch(
        read.csv(input$additional_train_file$datapath, header = TRUE),
        error = function(e) { NULL }
      )
      if (is.null(additional_df)) {
        showNotification("Error reading Additional New Training Data CSV.", type = "error")
        return()
      }
      if (!all(c("x", "y", "label") %in% names(additional_df))) {
        showNotification("Additional training data must have columns: x, y, label.", type = "error")
        return()
      }
      additional_df$x <- round(as.numeric(additional_df$x), 3)
      additional_df$y <- round(as.numeric(additional_df$y), 3)
      additional_df$label <- as.factor(additional_df$label)
    }
    
    if (!is.null(additional_df)) {
      combined_new <- rbind(new_labels, additional_df)
    } else {
      # Make sure new_labels has the same columns as trainingData
      combined_new <- new_labels[ , names(trainingData()), drop = FALSE]
    }
    
    # Merge new labeled samples into training data
    updated_train <- rbind(trainingData(), combined_new)
    trainingData(updated_train)
    updatedTrainData(updated_train)
    showNotification(paste("Updated training data with", nrow(combined_new), "new labeled samples."), type = "message")
    
    # Remove newly labeled samples from the unlabeled data
    if (!is.null(unlabeledData())) {
      ul <- unlabeledData()
      # Round x and y in unlabeled data to 3 decimals for matching
      ul$x <- round(as.numeric(ul$x), 3)
      ul$y <- round(as.numeric(ul$y), 3)
      new_ids <- paste(combined_new$x, combined_new$y, sep = "_")
      ul_ids <- paste(ul$x, ul$y, sep = "_")
      updated_unlabeled <- ul[!(ul_ids %in% new_ids), ]
      unlabeledData(updated_unlabeled)
      showNotification("Removed newly labeled samples from unlabeled data.", type = "message")
    }
  })
  
  # # 5. Retrain Model using updated training data
  # observeEvent(input$retrain_model, {
  #   req(trainingData())
  #   model_method <- switch(
  #     input$model_type,
  #     "SVM" = "svmRadial",
  #     "Random Forest" = "rf",
  #     "ANN" = "nnet",
  #     "GAM" = "gam"
  #   )
  #   cv_ctrl <- trainControl(method = "cv", number = input$cv_folds,
  #                           repeats = if (input$cv_repeats > 1) input$cv_repeats else 1)
  #   withProgress(message = "Retraining model on updated dataset...", {
  #     model <- train(label ~ ., data = trainingData(),
  #                    method = model_method,
  #                    trControl = cv_ctrl,
  #                    tuneLength = input$tune_length)
  #     trainedModel(model)
  #   })
  #   output$train_results <- renderPrint({ trainedModel() })
  #   showNotification("Model retrained with updated training data.", type = "message")
  #   
  #   if (!is.null(unlabeledData())) {
  #     probs <- predict(trainedModel(), newdata = unlabeledData(), type = "prob")
  #     preds <- predict(trainedModel(), newdata = unlabeledData(), type = "raw")
  #     if (all(c("1", "0") %in% names(probs))) {
  #       names(probs)[names(probs) == "1"] <- "landslide"
  #       names(probs)[names(probs) == "0"] <- "nonlandslide"
  #     }
  #     result_df <- cbind(unlabeledData(), PredictedLabel = preds, probs)
  #     result_df <- unique(result_df)
  #     predictions(result_df)
  #     output$predictions_table <- renderTable({ head(, 10) }) #renderTable({result_df})
  #   }
  #   uncertainSamples(NULL)
  # })
  
  # 5. Download Handlers
  output$download_predictions <- downloadHandler(
    filename = function() { "model_predictions.csv" },
    content = function(file) {
      req(predictions())
      write.csv(predictions(), file, row.names = FALSE)
    }
  )
  
  output$download_training <- downloadHandler(
    filename = function() { "training_samples.csv" },
    content = function(file) {
      req(trainingData())
      write.csv(trainingData(), file, row.names = FALSE)
    }
  )
  
  # Display updated training data preview (first 10 rows)
  output$updated_train_table <- renderPrint({
    req(updatedTrainData())
    str(updatedTrainData())#, 10
  })
}

shinyApp(ui = ui, server = server)
