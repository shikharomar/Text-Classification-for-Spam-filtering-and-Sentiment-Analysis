# loading sms data from the csv file
sms_raw <- read.csv('sms_spam.csv', stringsAsFactors = FALSE)

# getting a glimpse of the data
head(sms_raw)
str(sms_raw)

# converting labels as factor(categorical variable)
sms_raw$type <- as.factor(sms_raw$type)

# getting the structure of the labels and their proportion in the dataset
table(sms_raw$type)

# loading text mining (tm) package of R
library('tm')

# creating corpus out of the sms data
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

# examining the corpus
inspect(sms_corpus[1:2])
as.character(sms_corpus[[1]])
lapply(sms_corpus[1:3], as.character)

# loading package for using pipeline
library(magrittr)

# data cleaning 
sms_corpus_cleaned <- sms_corpus %>% tm_map(content_transformer(tolower)) %>% 
  tm_map(removeWords, stopwords(kind = 'en')) %>% 
  tm_map(removeNumbers) %>% 
  tm_map(removePunctuation) %>% 
  tm_map(stemDocument) %>%  # using SnowballC package
  tm_map(stripWhitespace)

# creating data term matrix after data cleaning
sms_dtm <- DocumentTermMatrix(sms_corpus_cleaned)

# creating data term matrix along with data cleaning
sms_dtm2 <- DocumentTermMatrix(sms_corpus, control = list(tolower = TRUE, removeNumbers = TRUE,
                                                          stopwords = TRUE, removePunctuation = TRUE,
                                                          stemming = TRUE, stripWhitespace = TRUE))

# comparing the results 
print(sms_dtm)
print(sms_dtm2)

# data splitting for training and testing
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]

# getting the lables for the training and testing data
sms_test_labels <- sms_raw[4170: 5559, ]$type
sms_train_labels <- sms_raw[1:4169, ]$type

# checking whether both data(train and test) represent the whole dataset
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

# creating word cloud for the most frequent words
library(wordcloud)
wordcloud(sms_corpus_cleaned, min.freq = 100, random.order = FALSE)

# subsetting train and test data for word cloud visualization
spam <- subset(sms_raw, type == 'spam')
ham <- subset(sms_raw, type == 'ham')

wordcloud(spam$text, max.words = 40, random.order = FALSE, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, random.order = FALSE, scale = c(3, 0.5))

# reducing the size of the feature set
sms_freq_terms <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_terms)

# selecting only frequent features for the training data
sms_dtm_freq_train<- sms_dtm_train[ , sms_freq_terms]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_terms]

# converting occurence and non-occurence into binary values (emphasizing occurence instead of frequency)
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

# loading library for Naive Bayes classifier and training the model
library(e1071)
NB_classifier <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
prediction <-  predict(NB_classifier, sms_test, type = 'class')

# examining the performance of the classifier
library(gmodels)
CrossTable(prediction, sms_test_labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c('Prediction', 'Actual'))
print(sum(prediction == sms_test_labels)/length(prediction)) # 97.63%

# reference: Machine Learning with R , Brett Lantz
