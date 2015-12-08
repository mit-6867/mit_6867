library(gender)

first_name <- function(x) {
  split <- strsplit(x, ' ')
  first_name <- split[[1]][1]
  return (first_name)
}

male_prop_for_df <- function(x) {
  predicted_gender <- gender(x[['first_name']], years = c(1935, 1997), method="ssa")
  return(predicted_gender$proportion_male)
}

female_prop_for_df <- function(x) {
  predicted_gender <- gender(x[['first_name']], years = c(1935, 1997), method="ssa")
  return(predicted_gender$proportion_female)
}

authors <- read.delim('author_list.tsv', head=T)
authors$author_name <- as.character(authors$author_name)
authors$first_name <- apply(authors, 1, FUN=first_name)
authors$prop_male <- apply(authors, 1, FUN=male_prop_for_df)
authors$prop_female <- apply(authors, 1, FUN=female_prop_for_df)

authors$prop_male <- as.numeric(authors$prop_male)
authors$prop_female <- as.numeric(authors$prop_female)

write.table(authors, file='authors_with_genders.tsv', row.names=FALSE, sep="\t", 
            quote=FALSE, fileEncoding='UTF-8')


# split the name string to get multiple author names 

n <- str_split(authors$names,'AND ')

authors_number <- vector()

for (i in 1:7367){
  print (i)
  authors_number[i] <- length(n[[i]])
}
max(authors_number)

author_1_first_name <- vector()
author_2_first_name <- vector()

for (i in 1:7367){
  print (i)
  if(n[[i]]==''){
    author_1_first_name[i] = 'NA'
    author_2_first_name[i] = 'NA'
  } else if (length(n[[i]])==1){
      author_1_first_name[i] = first_name(n[[i]])
      author_2_first_name[i] = 'NA'
    } else {
    author_1_first_name[i] = first_name(n[[i]][1])
    author_2_first_name[i] = first_name(n[[i]][2])
  }
}

authors <- cbind(authors,author_1_first_name,author_2_first_name)

colnames(authors)[5] <- 'prop_male_1'
colnames(authors)[6] <- 'prop_female_1'

critical <- .5
authors$gender_1 <- rep('NA',nrow(authors))
authors$gender_2 <- rep('NA',nrow(authors))

authors$gender_1[which(authors$prop_male_1=='NA')] <- 'NA'
authors$gender_1[which(authors$prop_male_1!='NA' & authors$prop_male_1>=critical)] <- '0'
authors$gender_1[which(authors$prop_male_1!='NA' & authors$prop_female_1 >= critical)] <- '1'

authors$gender_2[which(authors$prop_male_2=='NA')] <- 'NA'
authors$gender_2[which(authors$prop_male_2!='NA' & authors$prop_male_2>=critical)] <- '0'
authors$gender_2[which(authors$prop_male_2!='NA' & authors$prop_female_2 >= critical)] <- '1'

table(authors$gender_1)
table(authors$gender_2)

write.csv(authors,'multiple_authors.csv',row.names = F)
