library(stringr)
library(dplyr)

text <- data.frame(index = c(), text=c(), stringsAsFactors=FALSE)
for (i in 1:7368){
  print(i)
  name <- paste('../../text_scraping/article_text/', i,'.txt',sep='')
  t <- tryCatch({
    readChar(name, file.info(name)$size)
  }, warning=function(w) {
    return('NA')
  }, error=function(e) {
    return('NA')
  }, finally = {
  })
  t <- str_replace_all(t, '\n', '')
  text = rbind(text, c(i, t))
  names(text) <- c('index', 'text')
  text$index <- as.numeric(text$index)
  text$text <- as.character(text$text)
}

write.csv(text,'articles.csv',row.names=FALSE)

nonempty_articles <- filter(text,text!='')
set.seed(111)
s <- sample(nonempty_articles$index,200)
sampled_articles <- nonempty_articles[s,]

head(sampled_articles)
sampled_articles %>%
  mutate(text = str_replace_all(text, ',', '')) -> sampled_articles

write.csv(dplyr::select(sampled_articles, text),'sample_articles.csv',row.names=FALSE, fileEncoding='utf-8', quote=FALSE)
