library(ggplot2)
library(knitr)
library(dplyr)
library(tidyr)
library(stringr)
library(scales)
setwd('/Users/dholtz/Desktop/mit_6867/Project/Latex/')

weights <- c(-0.00623197060315, -0.0242271706056, -0.014351445239, 0.0441530957983, -0.183141438435, -0.070792093083, 0.0479733002858, 0.0356982720475, 0.0162439138474, 0.579854386616, 0.120247858437, 0.1171156196, 0.0942094479819, -0.105564058985, -0.00421410048242, 0.278003363208, 0.0363435856959, -0.0859070347175, -0.109571640119, 0.0640607591967, 0.0346550859402, -0.240917647691, 0.0276850189048, -0.186985947534, -0.204141702973, 0.0448847885443, -0.482790157831, 0.00272621354153, -0.106190380173, 0.00398383468196, -0.101871591649, 0.733401300913, 0.0193655004814, -0.0327953544567, 0.0354846836801, 0.235394877441, 0.135712802758, -0.0316455913835, -0.125844772176, -0.144438638638, 0.047973300281, -0.14405519998, 0.349880569091, 0.00528367806774, 0.192871661011, -0.103270596605, -0.0124195747141, -0.0493038049738, -0.268649476666, 0.00678743426196, 0.135766399505, 0.127683656849, 0.238839384074, -0.0196807004655, -0.0568528106611, 0.0224381468046, -0.209821773103, 0.0440742363626, 0.051248749088, 0.0710445438536, 0.092186731318, -0.0869283836069, -0.0537041550833, -0.115784992925, -0.452823135391, -0.00771210916169, 0.0103434730221, -0.0249938994228, 0.260697275868, 0.152143644462, 0.19517491076, -0.0110564008367, 0.0267366730432, -0.11583385892, -0.000500866474, 0.00600399969973, 0.0040386595581, -0.294996109746, -0.0368505591042, 0.022672483996, 0.0541801991396, 0.0592330487156, -0.000601681234961, 0.0593623261442, 0.190843349729, -0.0418335251941, 0.0533460621188, -0.0318690621793, -0.302743426906, -0.319668694839, -0.0148220651828, -0.0130394903733, -0.0917688101034, -0.00721212406133, -0.516239251888, 0.0266169956416, 0.0631139305705, 0.00850334672107, -0.00545365435006, 0.0555282923315, 0.0129282301541, 9.12145353103)
names <- c("author_gender_ambiguous / unknown", "author_gender_female", "desk_Arts&Leisure", "desk_Automobiles", "desk_BookReview", "desk_Culture", "desk_Dining", "desk_EdLife", "desk_Editorial", "desk_Foreign", "desk_Home", "desk_Letters", "desk_Magazine", "desk_Metro", "desk_NODESK", "desk_National", "desk_NewsDesk", "desk_None", "desk_OpEd", "desk_RealEstate", "desk_Science", "desk_Society", "desk_Sports", "desk_Styles", "desk_Summary", "desk_SundayBusiness", "desk_Travel", "desk_Washington", "desk_Weekend", "flesch-kincaid_score", "log_popularity", "log_wcount", "perplexity", "section_Automobiles", "section_Blogs", "section_Books", "section_Booming", "section_Business Day", "section_Corrections", "section_Crosswords & Games", "section_Dining & Wine", "section_Education", "section_Fashion & Style", "section_Great Homes and Destinations", "section_Health", "section_Home & Garden", "section_Job Market", "section_Magazine", "section_Movies", "section_Multimedia", "section_Multimedia/Photos", "section_N.Y. / Region", "section_Opinion", "section_Public Editor", "section_Real Estate", "section_Science", "section_Sports", "section_Style", "section_Sunday Review", "section_T Magazine", "section_Technology", "section_Theater", "section_Travel", "section_U.S.", "section_World", "section_Your Money", "sentiment_negative", "sentiment_positive", "timeofday_12-17", "timeofday_18-23", "timeofday_6-11", "typeOfMaterial_An Appraisal", "typeOfMaterial_Biography", "typeOfMaterial_Brief", "typeOfMaterial_Correction", "typeOfMaterial_Economic Analysis", "typeOfMaterial_Editorial", "typeOfMaterial_Letter", "typeOfMaterial_List", "typeOfMaterial_Military Analysis", "typeOfMaterial_News", "typeOfMaterial_News Analysis", "typeOfMaterial_Obituary", "typeOfMaterial_Obituary (Obit)", "typeOfMaterial_Op-Ed", "typeOfMaterial_Question", "typeOfMaterial_Quote", "typeOfMaterial_Recipe", "typeOfMaterial_Review", "typeOfMaterial_Schedule", "typeOfMaterial_Series", "typeOfMaterial_Summary", "typeOfMaterial_Text", "typeOfMaterial_Web Log", "type_BlogPost", "weekday_Monday", "weekday_Saturday", "weekday_Sunday", "weekday_Thursday", "weekday_Tuesday", "weekday_Wednesday", "intercept")

linear_regression_weights = data.frame(weights = weights, names = names)

linear_regression_weights %>% 
  mutate(names = str_replace_all(str_replace_all(str_replace_all(str_replace_all(
    str_replace_all(
      str_replace_all(
        str_replace_all(
          str_replace_all(
            str_replace_all(
              str_replace_all(
                str_replace_all(names, 'author_gender_', 'Author Gender: '),
        'desk_', 'Desk: ')
    , 'section_', 'Section: '), 'timeofday_', 'Time of Day: '),
    'typeOfMaterial_', 'Type of Material: '), 'type_', 'Type: '),
    'weekday_', 'Day of Week: '), 'flesch-kincaid_score', 'Flesch Reading Ease'), 
    'log_wcount', 'log(Word Count)'), 'log_popularity', 'log(Popularity)'), 'sentiment_', 'Sentiment: ')
    ) -> linear_regression_weights

ggplot(filter(linear_regression_weights, names!='intercept'), aes(x=names, y=weights)) + 
  geom_bar(stat="identity", position="dodge") + theme(axis.text.x=element_text(angle=90, size=5)) + 
  xlab('Feature') + ylab('Feature Weight') + ggtitle('Linear Regression Feature Weights') -> p
ggsave(p, file='feature_weights.png')

linear_regression_weights %>%
  arrange(desc(abs(weights))) %>%
  mutate(weights = round(weights, 3)) %>%
  dplyr::select(names, weights) %>%
  head(n=50) %>%
  kable(., format='latex')

mse_performance <- read.csv('~/Desktop/mit_6867/Project/regression/mse_performance.csv')

mse_performance %>%
  gather(group, mse, training_mse:holdout_mse) %>%
  mutate(group = ifelse(group == 'training_mse', 'Training', 'Validation')) %>%
  ggplot(., aes(x=lambdas, y=mse, color=group)) + geom_point(size=2) + 
  xlab('Regularization Parameter') + ylab('MSE') + 
  ggtitle('Effect of Regularization on MSE') -> p2

ggsave(p2, file='mse_plot.png')

author_data <- read.delim('../gender_guessing/authors_with_genders.tsv', sep='\t')

ggplot(author_data, aes(x=popularity)) + geom_histogram() + 
  scale_x_continuous(labels=comma, lim=c(0, 100000)) + 
  xlab('Number of Bing Search Results') + ylab('count') -> p3
ggsave(p3, file='author_popularity_histogram.png')

ggplot(author_data, aes(x=log(popularity))) + geom_histogram() + 
  scale_x_continuous(labels=comma, lim=c(0, 25)) + 
  xlab('log(Number of Bing Search Results)') + ylab('count') -> p4
ggsave(p4, file='author_popularity_log_histogram.png')

flesch_data <- read.csv('../feature_extraction/readability/flesch_kincaid.txt')

ggplot(flesch_data, aes(x=flesch.kincaid_score)) + geom_histogram() +
  xlim(c(0, 120)) + xlab('Flesch Reading Ease') + ylab('count') -> p5
ggsave(p5, file='flesch_data_histogram.png')

bigram_plot <- read.csv('../regression/bigram_plot_df.csv')

bigram_plot %>% 
  mutate(value = -value) %>%
ggplot(., aes(x=alpha, y=value, color=variable)) + geom_line() + 
  xlab('Alpha') + ylab('Average log likelihood per word') + 
  theme(legend.position='bottom') -> p6
ggsave(p6, file='bigram_plot_df.png')
