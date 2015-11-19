# install packages and define parse function

if (!require('XML')) {install.packages('XML')}
if (!require('httr')) {install.packages('httr')}

# old parse function, article only

parseArticleBody <- function(artHTML) {
  xpath2try <- c('//div[@class="articleBody"]//p',
                 '//p[@class="story-body-text story-content"]',
                 '//p[@class="story-body-text"]'
  )
  for(xp in xpath2try) {
    bodyi <- paste(xpathSApply(htmlParse(artHTML), xp, xmlValue), collapse='')
    if(nchar(bodyi)>0) break
  }
  return(bodyi)
}

# new parse function, dummy for video and interactive graph

parseArticleBody <- function(artHTML) {
  xpath2try <- c('//div[@class="articleBody"]//p',
                 '//p[@class="story-body-text story-content"]',
                 '//p[@class="story-body-text"]'
  )
  
  for(xp in xpath2try) {
    bodyi <- paste(xpathSApply(htmlParse(artHTML), xp, xmlValue), collapse='')
    if(nchar(bodyi)>0) break
  }
  
  xpath <- c('//a[@class="video-link"]')
  video <- paste(xpathSApply(htmlParse(artHTML), xpath, xmlValue), collapse='')
  if (nchar(video)!= 0) {v <- 1} else {v <- 0}
  
  ypath <- c('//div[@class="interactive-graphic"]')
  graph <- paste(xpathSApply(htmlParse(artHTML), ypath, xmlValue), collapse='')
  if (nchar(graph)!= 0) {g <- 1} else {g <- 0}
  
  return(list(bodyi,v,g))
}

path_to_article_list <- 'article_list.txt'
#path_to_article_list <- '/srv/ml_project/article_list/9-list'


article_list <- read.csv(path_to_article_list, header=F)
names(article_list) <- c('article_url')

article_dummies <- data.frame(index = c(), 
  has_video = c(),
  has_graph = c()
  )

for (i in 1:length(article_list$article_url)) {
  print(i)
  url <- GET(as.character(article_list$article_url[i]))
  html <- content(url, 'text')
  artBody <- parseArticleBody(html)
  write(artBody[[1]], file=paste0('article_text/', as.character(i), '.txt', sep="", collapse=""))
  article_dummies <- rbind(article_dummies, c(i, artBody[[2]], artBody[[3]]))
  names(article_dummies) <- c('index', 'has_video', 'has_graph')
  write.table(article_dummies, file='article_dummies.txt', quote=FALSE, row.names=F, col.names=T, sep=',')
}

# Example: extract article text for one article

#url <- GET('http://www.nytimes.com/2015/11/16/world/europe/inquiry-finds-mounting-proof-of-syria-link-to-paris-attacks.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=a-lede-package-region&region=top-news&WT.nav=top-news')
#url <- GET('http://dotearth.blogs.nytimes.com/2013/08/01/google-science-fellows-challenge-companys-support-for-inhof/')

#html <- content(url, 'text')
#artBody <- parseArticleBody(html)
#print(artBody)
