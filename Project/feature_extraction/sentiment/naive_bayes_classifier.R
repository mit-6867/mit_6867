
library('stringr')
library('tm')
library('SnowballC')
library('slam')
#library("tm.lexicon.GeneralInquirer")
library('readr')
library('dplyr')
#library('e1071')

# read in training data
training <- read.csv('training.csv',encoding='UTF-8')
head(training)
tail(training)
dim(training)

mean(training$Avg)
table(training$Avg)
table(training$Answer1)

training$Label <- rep('NA',200)
training$Label[which(training$Avg >= .6)] <- 1
training$Label[which(training$Avg <= -.6)] <- -1
training$Label[which(training$Avg > -.6 & training$Avg < .6)] <- 0

pos = filter(training,Label=='1')
neg = filter(training,Label=='-1')
neu = filter(training,Label=='0')

dim(pos)
dim(neg)
dim(neu)

head(training)


normalize <- function(corpus){ # conversion to lower case, other function: removeNumbers,removePunctuation,removeWords, 
  wd <- tm_map(corpus,content_transformer(removePunctuation))
  wd <- tm_map(wd,content_transformer(tolower)) 
  wd <- tm_map(wd,content_transformer(removeNumbers))
  wd <- tm_map(wd,content_transformer(removeWords),stopwords("english"))
  #  wd <- tm_map(wd,content_transformer(removeWords),c('may','will'))
  wd <- tm_map(wd,content_transformer(stripWhitespace))
  wd <- tm_map(wd,content_transformer(stemDocument))
  return(wd)}

#c <- Corpus(DirSource('C:/Users/Jeremy/Desktop/Data/R/articles',encoding='UTF-8'))

text <- training$text # all classes

# separate for each class
text <- pos$text
text <- neg$text
text <- neu$text

c <- Corpus(VectorSource(text))
c <- normalize(c)
dtm <- DocumentTermMatrix(c)

dim(dtm)

# number of words for each doc
len <- rollup(dtm, 2, na.rm=TRUE, FUN = sum)
len <- as.vector(as.matrix(len))
length(len)

# number of each words in a given class
wordcount <- rollup(dtm, 1, na.rm=TRUE, FUN = sum)
wordcount <- as.vector(as.matrix(wordcount))
length(wordcount)

wordlist <- dtm$dimnames[[2]]
wordlist[1:10]
length(wordlist)

wordlist_pos = wordlist
wordlist_neg = wordlist 
wordlist_neu = wordlist

len_pos = len
len_neg = len
len_neu = len

wordcount_all = 18349

prob_condition_on_pos <- as.numeric((wordcount + 1)/(sum(len) + wordcount_all))

prob_condition_on_neg <- as.numeric((wordcount + 1)/(sum(len) + wordcount_all))

prob_condition_on_neu <- as.numeric((wordcount + 1)/(sum(len) + wordcount_all))

dic_pos = as.data.frame(cbind(wordlist,prob_condition_on_pos))

dic_neg = as.data.frame(cbind(wordlist,prob_condition_on_neg))

dic_neu = as.data.frame(cbind(wordlist,prob_condition_on_neu))

#wordlist = str_replace_all(wordlist,'\"','')
#wordlist = str_replace_all(wordlist,"'",'')

#wordlist = gsub("'",'',wordlist)

head(dic_neg)
tail(dic_neg)
tail(dic_pos)

# prediction, new wordlist 

prob_pos
prob_neg 
prob_neu

test <- readLines('873.txt',encoding='UTF-8')
test <- str_replace_all(test,"[[:punct:]|[0-9]]",'')
test <- str_to_lower(test)
word <- str_split(test,' ')[[1]]

zero_pos <- as.numeric((1)/(sum(len_pos) + wordcount_all))
zero_neg <- as.numeric((1)/(sum(len_neg) + wordcount_all))
zero_neu <- as.numeric((1)/(sum(len_neu) + wordcount_all))

seudo_prob_pos <- vector()
seudo_prob_neg <- vector()
seudo_prob_neu <- vector()

pred <- function (word){
  
  for (w in word){
    
    if (length(as.character(filter(dic_pos, wordlist==w)[,2]))!=0){
      seudo_prob_pos[w] = as.character(filter(dic_pos, wordlist==w)[,2])
      
    } else {
      seudo_prob_pos[w] = zero_pos
    }
    
    if (length(as.character(filter(dic_neg, wordlist==w)[,2]))!=0){
      seudo_prob_neg[w] = as.character(filter(dic_neg, wordlist==w)[,2])
      
    } else {
      seudo_prob_neg[w] = zero_neg
    }
    
    if (length(as.character(filter(dic_neu, wordlist==w)[,2]))!=0){
      seudo_prob_neu[w] = as.character(filter(dic_neu, wordlist==w)[,2])
      
    } else {
      seudo_prob_neu[w] = zero_neu
    }
    
  }
  
  seudo_prob_pos <- as.numeric(seudo_prob_pos)
  seudo_prob_neg <- as.numeric(seudo_prob_neg)
  seudo_prob_neu <- as.numeric(seudo_prob_neu)
  
  if (max(sum(seudo_prob_pos),sum(seudo_prob_neg),sum(seudo_prob_neu)) == sum(seudo_prob_pos)){
    class <- 1
  } else if (max(sum(seudo_prob_pos),sum(seudo_prob_neg),sum(seudo_prob_neu)) == sum(seudo_prob_neg)
  ) {
    class <- -1
  } else {
    class <- 0
  }
  
  return(class)
  
}

pred(word)

n <- 10
n <- 6682

text[1]

sentiment <- vector()

text <- as.character(text)

for (i in 1:50){
 
 test <- str_replace_all(text[i],"[[:punct:]|[0-9]]",'')
 test <- str_to_lower(test)
 word <- str_split(test,' ')[[1]]
 
 sentiment[i] <- pred(word)
 print (i)

}

test <- str_replace_all(text[100],"[[:punct:]|[0-9]]",'')
test <- str_to_lower(test)
word <- str_split(test,' ')[[1]]
pred(word)

sentiment



