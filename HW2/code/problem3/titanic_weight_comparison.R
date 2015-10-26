feature_weights = data.frame(`Model` = vector(mode='character', length=44))

feature_weights[['Model']] = c(rep('Logistic Regression', 11), 
                               rep('Linear SVM', 11), 
                               rep('Polynomial SVM', 11), 
                               rep('RBF SVM', 11))

feature_names = c('Passenger Class = 1st', 'Passenger Class = 2nd', 
                  'Passenger Class = 3rd', 'Gender', 'Age', '# Siblings/Spouses', 
                  '# Parents/Children', 'Passenger Fare', 'Embarked at Southhampton', 
                  'Embarked at Cherbourg', 'Embarked at Queenstown')
feature_weights[['Feature Name']] = rep(feature_names, 4)

weights = c(0.6036931,0.45528582,-1.05897834,2.47558843,-0.91435904,-0.251056,
            0.53815509,0.33064367,-0.2331139,0.29161072,-0.05849566, 0.0313021,
            0.0339519,-0.06524453,1.90113447,-0.07850197,-0.01150343,0.13105129,
            0.0577874,-0.03825494,0.02932354,0.00894087, -0.01289241,-0.01050105,
            0.02351099,0.10193199,-0.00904929,0.00016298,0.01068262,-0.00154886,
            -0.00986365,0.0248439,-0.01486273, 0.12005924,0.25510333,-0.37513945,
            1.44000021,-0.06826854,-0.03664755,0.1049032,0.11751473,-0.15997783,
            0.24000049,-0.07999953)

feature_weights[['Feature Weight']] = weights

feature_weights %>%
  group_by(`Model`) %>%
  mutate(max_weight = max(abs(`Feature Weight`))) %>%
  ungroup() %>% 
  mutate(`Feature Weight (Normalized)` = `Feature Weight`/max_weight) %>%
  ggplot(., aes(x=as.factor(`Feature Name`), y=`Feature Weight (Normalized)`, fill=`Model`)) + 
  geom_bar(stat="identity", position="dodge") + xlab('Feature') + 
  ggtitle('Feature Weights (Normalized by Model)') + theme(axis.text.x = element_text(angle = 90, hjust = 1)) -> p
ggsave(p, file='~/Desktop/mit_6867/HW2/tex/titanic_weight_comparison.png')

feature_weights %>%
  group_by(`Model`) %>%
  mutate(max_weight = max(abs(`Feature Weight`))) %>%
  ungroup() %>% 
  mutate(`Feature Weight (Normalized)` = `Feature Weight`/max_weight) %>%
  ggplot(., aes(x=as.factor(`Feature Name`), y=`Feature Weight`, fill=`Model`)) + 
  geom_bar(stat="identity", position="dodge")
