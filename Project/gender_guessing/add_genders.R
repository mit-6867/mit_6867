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

write.csv(authors, file='authors_with_genders.csv')

