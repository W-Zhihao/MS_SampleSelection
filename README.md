# MS_SampleSelection

MS_SampleSelection is an R Shiny application that provides a graphical user interface for an **Active Learning** tool using a **Margin Sampling** strategy. It allows users to interactively train a machine learning model (Support Vector Machine, Random Forest, Artificial Neural Network, or Generalized Additive Model) on an initial labeled dataset, query the most **uncertain unlabeled samples** for labeling, incorporate new labels, and iterate – all through an easy-to-use GUI. This tool is useful for researchers who want to reduce labeling effort by focusing on the most informative samples.

## Purpose

The purpose of this tool is to demonstrate and facilitate **pool-based active learning**. Instead of labeling all data, the model identifies which unlabeled data samples would be most informative if we knew their labels. By labeling those and adding them to the training set, the model can improve faster with fewer labels. **Margin sampling** is the strategy used to choose those informative points – it targets samples where the model is least confident (i.e., the predicted probabilities for the top two classes are very close, indicating ambiguity). This Shiny app lets users explore this process step by step with their own data or example data.

## Installation and Setup

**Requirements:** R (version 4.0+ recommended) and the following R packages:
- **shiny** for the web application framework  
- **caret** for training models with cross-validation  
- **randomForest** for Random Forest models  
- **nnet** for neural network (ANN) models  
- **mgcv** for generalized additive models (GAM)  
- **e1071** (or **kernlab**) for SVM (caret will use `kernlab` for SVM internally)  
- **DT** for interactive data tables in the app  

Install the packages in R if you don’t have them:  
```r
install.packages(c("shiny", "caret", "randomForest", "nnet", "mgcv", "e1071", "DT"))
