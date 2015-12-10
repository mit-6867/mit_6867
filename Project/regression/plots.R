library(ggplot2)
library(knitr)
library(dplyr)
library(tidyr)
library(stringr)
library(scales)
setwd('/Users/dholtz/Desktop/mit_6867/Project/Latex/')

weights <- c(-0.00935683780123, 0.0012888842884, -0.00176902964024, -0.0326410018745, -0.00873301996587, 0.0452225178536, -0.153963343222, -0.0513280585607, 0.0465783104632, 0.0285987258812, 0.0741204302159, 0.550357284095, 0.0652518817721, 0.0618039304425, 0.0863072144038, -0.0787043800792, 0.0139822648931, 0.21108173792, 0.00460087796063, -0.261501864262, -0.0531943262649, 0.0238684263565, 0.0422406755963, -0.0169602645735, 0.0133279633935, 0.04485410416, -0.152841258541, 0.0402789285297, -0.464533313506, 0.00110117195978, -0.113912714784, 0.00293462565198, -0.100062922372, 0.740666349519, -0.0228025341519, 0.00867548799328, 0.201124757439, 0.0177764542244, -0.00778336263491, -0.1481996739, -0.136842227045, 0.0465783104633, -0.124329887129, 0.0568752713603, 0.0302594979509, 0.21027044512, -0.040466730142, -0.0120733795062, -0.0478114193782, -0.239090948548, -0.109664016828, 0.139283598171, 0.115117666264, 0.283125015706, -0.0174273179981, -0.0173927039611, 0.0425052964862, -0.175093487863, 0.0510497820267, 0.0523228456723, 0.0799899743335, 0.10226459856, -0.0928221284121, -0.056757160699, -0.0276612829879, -0.402025577332, -0.00395153082018, 0.00564223298758, -0.0306027758511, 0.23233371491, 0.135096054435, 0.146477474781, -0.00772850272177, 0.0279941730444, -0.0506773162729, -0.00501059606018, 0.0108622980696, -0.00953062930038, -0.19978081622, -0.0185668929728, 0.02633870046, 0.18741502583, 0.0470280359881, 0.00444474111097, 0.0891350513662, 0.170300922084, -0.00726113677605, 0.0530373017884, -0.0222696646608, -0.217762229432, -0.266502751939, 0.00687561246084, -0.0100993715467, -0.0917629992742, -0.00086101942235, -0.281818143357, 0.0350296235912, 0.0625763162511, 0.0259204152956, -0.00794888330179, 0.0475415200623, 0.020095195525, 8.96364989258)
names <- c("Squared Trigram Perplexity", "Trigram Perplexity", "author_gender_ambiguous / unknown", "author_gender_female", "desk_Arts&Leisure", "desk_Automobiles", "desk_BookReview", "desk_Culture", "desk_Dining", "desk_EdLife", "desk_Editorial", "desk_Foreign", "desk_Home", "desk_Letters", "desk_Magazine", "desk_Metro", "desk_NODESK", "desk_National", "desk_NewsDesk", "desk_None", "desk_OpEd", "desk_RealEstate", "desk_Science", "desk_Society", "desk_Sports", "desk_Styles", "desk_Summary", "desk_SundayBusiness", "desk_Travel", "desk_Washington", "desk_Weekend", "flesch-kincaid_score", "log_popularity", "log_wcount", "section_Automobiles", "section_Blogs", "section_Books", "section_Booming", "section_Business Day", "section_Corrections", "section_Crosswords & Games", "section_Dining & Wine", "section_Education", "section_Fashion & Style", "section_Great Homes and Destinations", "section_Health", "section_Home & Garden", "section_Job Market", "section_Magazine", "section_Movies", "section_Multimedia", "section_Multimedia/Photos", "section_N.Y. / Region", "section_Opinion", "section_Public Editor", "section_Real Estate", "section_Science", "section_Sports", "section_Style", "section_Sunday Review", "section_T Magazine", "section_Technology", "section_Theater", "section_Travel", "section_U.S.", "section_World", "section_Your Money", "sentiment_negative", "sentiment_positive", "timeofday_12-17", "timeofday_18-23", "timeofday_6-11", "typeOfMaterial_An Appraisal", "typeOfMaterial_Biography", "typeOfMaterial_Brief", "typeOfMaterial_Correction", "typeOfMaterial_Economic Analysis", "typeOfMaterial_Editorial", "typeOfMaterial_Letter", "typeOfMaterial_List", "typeOfMaterial_Military Analysis", "typeOfMaterial_News", "typeOfMaterial_News Analysis", "typeOfMaterial_Obituary", "typeOfMaterial_Obituary (Obit)", "typeOfMaterial_Op-Ed", "typeOfMaterial_Question", "typeOfMaterial_Quote", "typeOfMaterial_Recipe", "typeOfMaterial_Review", "typeOfMaterial_Schedule", "typeOfMaterial_Series", "typeOfMaterial_Summary", "typeOfMaterial_Text", "typeOfMaterial_Web Log", "type_BlogPost", "weekday_Monday", "weekday_Saturday", "weekday_Sunday", "weekday_Thursday", "weekday_Tuesday", "weekday_Wednesday", "intercept")

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

ggplot(filter(linear_regression_weights[1:53,], names!='intercept'), aes(x=names, y=weights)) + 
  geom_bar(stat="identity", position="dodge") + theme(axis.text.x=element_text(angle=90, size=10)) + 
  xlab('Feature') + ylab('Feature Weight') + ggtitle('Linear Regression Feature Weights') -> p
ggsave(p, file='feature_weights.png')

ggplot(filter(linear_regression_weights[54:103,], names!='intercept'), aes(x=names, y=weights)) + 
  geom_bar(stat="identity", position="dodge") + theme(axis.text.x=element_text(angle=90, size=10)) + 
  xlab('Feature') + ylab('Feature Weight') + ggtitle('Linear Regression Feature Weights') -> pcont
ggsave(pcont, file='feature_weights_contd.png')

linear_regression_weights %>%
  arrange(desc(abs(weights))) %>%
  mutate(weights = round(weights, 3)) %>%
  dplyr::select(names, weights) %>%
  head(n=20) %>%
  kable(., format='latex')

mse_performance <- read.csv('~/Desktop/mit_6867/Project/regression/mse_performance.csv')

mse_performance %>%
  gather(group, mse, training_mse:test_mse) %>%
  mutate(`Dataset` = ifelse(group == 'training_mse', 'Training', 
                        ifelse(group == 'validation_mse', 'Validation', 'Test'))
         ) %>%
  ggplot(., aes(x=lambdas, y=mse, color=Dataset)) + geom_point(size=2) + 
  xlab('Regularization Parameter') + ylab('MSE') + 
  ggtitle('Effect of Regularization on MSE') -> p2

ggsave(p2, file='mse_plot.png', width = 8, height = 6)

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

naive_bayes <- read.csv('../feature_extraction/sentiment/naive_bayes_merged.csv')
