# loading all the required packages
library(tm)
library(doMC)

# using all the cores
registerDoMC(cores = detectCores())

# loading the data
df <- read.csv("data/Train", sep = '\t', quote = "", header = FALSE, stringsAsFactors = FALSE)

# getting the insight of the dataset
str(df)

# checking for the unique entries in the dataset
length(unique(df$V2))

#removing the duplicates
V1 <- c()
V2 <- c()
V2 <- df$V2[!duplicated(df$V2)]
V1 <- df$V1[!duplicated(df$V2)]

# creating the new dataset after removing the duplicates
df2 <- data.frame(class = 1:length(V1), data = 1:length(V1))
df2$class <- V1
df2$data <- V2
rm(df, V1, V2)
df <- df2
rm(df2)

# randomizing the dataset
set.seed(0)
df <- df[sample(nrow(df)), ]

# using bag-of-words model for feature extraction from the sentences
# (In this approach, order of the words and the grammar is  not considered while the multiplicity is preserved)
corpus <- Corpus(VectorSource(df$data))
inspect(corpus[1:10])

# cleaning the  data
# (Stop words , puctuations, white spaces, numbers are removed)
corpus_cleaned <- corpus %>% tm_map(content_transformer(tolower)) %>% 
                  tm_map(removeNumbers) %>% 
                  tm_map(removePunctuation) %>% 
                  tm_map(removeWords, stopwords(kind = 'en')) %>% 
                  tm_map(stripWhitespace)

# representing bags of words as Document Term Matrix
dtm <- DocumentTermMatrix(corpus_cleaned)

# spliting dataset for training and testing
df_train <- df[1:1100,]
df_test <- df[1101:1410,]

dtm_train <- dtm[1:1100,]
dtm_test <- dtm[1101:1410,]

corpus_cleaned_train <- corpus_cleaned[1:1100]
corpus_cleaned_test <- corpus_cleaned[1101:1410]

# feature selection 
# (selecting only those features which appeared in more than 3 reviews)
freq_terms <- findFreqTerms(dtm_train, 3)
length((freq_terms))

# Document Term Matrix using only frequent terms
dtm_train_NB <-  DocumentTermMatrix(corpus_cleaned_train, control = list(dictionary = freq_terms))
dtm_test_NB <-  DocumentTermMatrix(corpus_cleaned_test, control = list(dictionary = freq_terms))

# final dataset for Binarized Naive Bayes(word occurence is more important than word frequency)

# for converting occurences to presence or absence
convert_count <- function(x){
  y <- ifelse(x > 0, 1, 0)
  y <- factor(y, levels = c(0, 1), labels = c('NO', 'YES'))
  y
}

# final dataset for  training and testing
train <- apply(dtm_train_NB, 2, convert_count)
test <- apply(dtm_test_NB, 2, convert_count)

# training the model
system.time(classifier <- naiveBayes(train, as.factor(df_train$class), laplace = 1))

# predicting on the test dataset
system.time( pred <- predict(classifier, newdata=test) )

# confusion matrix

conf_matrix <- confusionMatrix(pred, df_test$class)










