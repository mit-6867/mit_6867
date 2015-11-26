library('tm')
library('SnowballC')
library('slam')
library("tm.lexicon.GeneralInquirer")

normalize <- function(corpus){ # conversion to lower case, other function: removeNumbers,removePunctuation,removeWords, 
  wd <- tm_map(corpus,content_transformer(removePunctuation))
  wd <- tm_map(wd,content_transformer(tolower)) 
  wd <- tm_map(wd,content_transformer(removeNumbers))
  #  wd <- tm_map(wd,content_transformer(removeWords),stopwords("english"))
  #  wd <- tm_map(wd,content_transformer(removeWords),c('may','will'))
  #  wd <- tm_map(wd,content_transformer(stripWhitespace))
  #  wd <- tm_map(wd,content_transformer(stemDocument))
  return(wd)}

path <- 'C:/Users/Jeremy/Desktop/Data/R/articles'
c <- Corpus(DirSource(path,encoding='UTF-8'))
#c <- Corpus(VectorSource(text))
c <- normalize(c)
dtm <- DocumentTermMatrix(c)

# tf-idf weighted dtm
dtm_weighted <- DocumentTermMatrix(c, control = list(weighting = weightTfIdf))
  
pos <- as.numeric(tm_term_score(dtm,terms_in_General_Inquirer_categories("Positiv"))) 
neg <- as.numeric(tm_term_score(dtm,terms_in_General_Inquirer_categories("Negativ"))) 
len <- rollup(dtm, 2, na.rm=TRUE, FUN = sum)
len <- as.vector(as.matrix(len))

index <- seq(1,length(len))
data <- cbind(index,pos,neg,len)
head(data)
dim(data)

write.table(data,'sentiment.txt',row.names = F,sep=',')
