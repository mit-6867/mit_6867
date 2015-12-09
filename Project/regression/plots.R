library(ggplot2)
library(knitr)
library(dplyr)
library(tidyr)
library(stringr)
library(scales)
setwd('/Users/dholtz/Desktop/mit_6867/Project/Latex/')

weights <- c(-0.00191432502654, -0.0245399612398, -0.0150996164294, 0.0440296750968, -0.167166767298, -0.0396022800434, 0.0513236665953, 0.0376881505235, 0.034420292747, 0.560583168704, 0.135760449819, 0.119415176685, 0.0976386412178, -0.109717531939, 0.000263172714949, 0.245491206079, 0.00520118309151, -0.165878067649, -0.0892505580431, 0.0776715694698, 0.0461934763289, -0.177856267015, 0.0250636545508, -0.123728689152, -0.146765225723, 0.0466043122084, -0.477516203847, 0.00122058106983, -0.124056938997, 0.00101097778077, -0.0958951336039, 0.735969046607, 0.0167414635067, -0.025959165804, 0.040895009896, 0.219165180421, 0.108392843047, -0.0230247100238, -0.167360937146, -0.143143446175, 0.0513236665944, -0.13857883042, 0.274650293308, -0.00203664259554, 0.20002254626, -0.110278613843, -0.015642686809, -0.0503790999032, -0.267093153066, 0.00630394482605, 0.13920691866, 0.142020776932, 0.244810158572, -0.0206936510913, -0.0641111765411, 0.0342515812982, -0.190922375506, 0.0507730247675, 0.0552535755133, 0.074599895454, 0.0932262989123, -0.0826574211138, -0.0541942203966, -0.0675520953361, -0.409982572288, -0.00532750280566, 0.238705410021, 0.156883824326, 0.163951257457, -0.0177637990337, 0.0234011367299, -0.153357482844, -0.0168654857936, 0.00441377186855, -0.0265800308445, -0.310952228957, -0.0486760700795, 0.0211400997113, 0.00150676823874, 0.0435852536984, -0.00317176562678, 0.0389422324357, 0.15528305309, -0.0603885589553, 0.0301666928586, -0.0452872719984, -0.323507092898, -0.329889081099, -0.00411448761229, -0.0133846358151, -0.0934607743805, -0.00825110975912, -0.394127027181, 0.0491173950399, 0.0599957920327, 0.0289952809853, 0.00144497100566, 0.0541630652262, 0.0187797252149, 9.11388341686)
names <- c("author_gender_ambiguous / unknown", "author_gender_female", "desk_Arts&Leisure", "desk_Automobiles", "desk_BookReview", "desk_Culture", "desk_Dining", "desk_EdLife", "desk_Editorial", "desk_Foreign", "desk_Home", "desk_Letters", "desk_Magazine", "desk_Metro", "desk_NODESK", "desk_National", "desk_NewsDesk", "desk_None", "desk_OpEd", "desk_RealEstate", "desk_Science", "desk_Society", "desk_Sports", "desk_Styles", "desk_Summary", "desk_SundayBusiness", "desk_Travel", "desk_Washington", "desk_Weekend", "flesch-kincaid_score", "log_popularity", "log_wcount", "perplexity", "section_Automobiles", "section_Blogs", "section_Books", "section_Booming", "section_Business Day", "section_Corrections", "section_Crosswords & Games", "section_Dining & Wine", "section_Education", "section_Fashion & Style", "section_Great Homes and Destinations", "section_Health", "section_Home & Garden", "section_Job Market", "section_Magazine", "section_Movies", "section_Multimedia", "section_Multimedia/Photos", "section_N.Y. / Region", "section_Opinion", "section_Public Editor", "section_Real Estate", "section_Science", "section_Sports", "section_Style", "section_Sunday Review", "section_T Magazine", "section_Technology", "section_Theater", "section_Travel", "section_U.S.", "section_World", "section_Your Money", "timeofday_12-17", "timeofday_18-23", "timeofday_6-11", "typeOfMaterial_An Appraisal", "typeOfMaterial_Biography", "typeOfMaterial_Brief", "typeOfMaterial_Correction", "typeOfMaterial_Economic Analysis", "typeOfMaterial_Editorial", "typeOfMaterial_Letter", "typeOfMaterial_List", "typeOfMaterial_Military Analysis", "typeOfMaterial_News", "typeOfMaterial_News Analysis", "typeOfMaterial_Obituary", "typeOfMaterial_Obituary (Obit)", "typeOfMaterial_Op-Ed", "typeOfMaterial_Question", "typeOfMaterial_Quote", "typeOfMaterial_Recipe", "typeOfMaterial_Review", "typeOfMaterial_Schedule", "typeOfMaterial_Series", "typeOfMaterial_Summary", "typeOfMaterial_Text", "typeOfMaterial_Web Log", "type_BlogPost", "weekday_Monday", "weekday_Saturday", "weekday_Sunday", "weekday_Thursday", "weekday_Tuesday", "weekday_Wednesday", "intercept")

linear_regression_weights = data.frame(weights = weights, names = names)

linear_regression_weights %>% 
  mutate(names = str_replace_all(str_replace_all(str_replace_all(
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
    'log_wcount', 'log(Word Count)'), 'log_popularity', 'log(Popularity)')
    ) -> linear_regression_weights

ggplot(filter(linear_regression_weights, names!='intercept'), aes(x=names, y=weights)) + 
  geom_bar(stat="identity", position="dodge") + theme(axis.text.x=element_text(angle=90, size=5)) + 
  xlab('Feature') + ylab('Feature Weight') + ggtitle('Linear Regression Feature Weights') -> p
ggsave(p, file='feature_weights.png')

linear_regression_weights %>%
  arrange(desc(abs(weights))) %>%
  mutate(weights = round(weights, 3)) %>%
  dplyr::select(names, weights) %>%
  head(n=20) %>%
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
