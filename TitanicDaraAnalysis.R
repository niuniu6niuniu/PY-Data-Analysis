# Load raw data
train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)

# Add a column to the test table
test.survived <- data.frame(Survived = rep("None", nrow(test)), test[,])

# Combine test.survived table with train table
data.combined <- rbind(train, test.survived)

# R data type
str(data.combined)

# Convert "Survived" and "pClass" to factor
data.combined$Survived <- as.factor(data.combined$Survived)
data.combined$Pclass <- as.factor(data.combined$Pclass)

# Take a look at gross survival rate
table(data.combined$Survived)

# Distribution of class
table(data.combined$Pclass)

# Load ggplot package for visualization
library(ggplot2)

# Hypothesis - Rich folks survived at a higher rate
# Convert to factor
train$Pclass <- as.factor(train$Pclass)
ggplot(train, aes(x = Pclass, fill = factor(Survived))) +
  geom_histogram(binwidth = 0.5) + 
  xlab("Pclass") +
  ylab("Total Count") +
  labs(fill = "Survived")

# Examine the first few names in the training set
head(as.character(train$Name))

# How many unique names are there across both train & test?
length(unique(as.character(data.combined$Name)))

# Two duplicate names
# First, get the duplicate names and store them as a vector
dup.names <- as.character(data.combined[duplicated(data.combined$Name), "Name"])
# data.combined[data.combined$Name %in% dup.names,]

# Next, take a look at the records in the combined data set
data.combined[which(data.combined$Name %in% dup.names),]


# What is up with the 'Miss.' and 'Mr.' thing?
library(stringr)

# Any correlation with other variables (e.g., sibsp)?
misses <- data.combined[which(str_detect(data.combined$Name, "Miss.")), ]
misses[1:5,]

# Hypothesis - Name titles correlate with age
mrses <- data.combined[which(str_detect(data.combined$Name, "Mrs.")), ]
mrses[1:5,]

# Check out males to see if pattern continues
males <- data.combined[which(train$Sex == "male"), ]
males[1:5,]

# Expand upon the relationship between 'Survived' and 'Pclass' by adding the new 'Title' variable to the
# data set and then explore a potential 3-dimensional relationship

# Create a utility function to help with title extraction
extractTitle <- function(Name){
  name <- as.character(Name)
  
  if (length(grep("Miss.", Name)) > 0) {
    return ("Miss.")
  } else if (length(grep("Master.", Name)) > 0) {
    return ("Master.")
  } else if (length(grep("Mrs.", Name)) > 0) {
    return ("Mrs.")
  } else if (length(grep("Mr.", Name)) > 0) {
    return ("Mr.")
  } else {
    return ("Other")
  }
}

titles <- NULL
for (i in 1:nrow(data.combined)) {
  titles <- c(titles, extractTitle(data.combined[i,"Name"]))
}
data.combined$Title <- as.factor(titles)

# Since we only have survived labels for the train set, 
# only use the first 891 rows
ggplot(data.combined[1:891,], aes(x = Title, fill = Survived)) +
  geom_bar(width = 0.5) +
  facet_wrap(~Pclass) +
  ggtitle("Pclass") +
  xlab("Title") +
  ylab("Total Count") +
  labs(fill = "Survived")
