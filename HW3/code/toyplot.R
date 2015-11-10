library(ggplot2)
library(dplyr)

toy_test_1 <- read.csv('~/Downloads/toy_multiclass_1_testCLEAN.csv', head=T)
toy_validate_1 <- read.csv('~/Downloads/toy_multiclass_1_validateCLEAN.csv', head=T)
toy_train_1 <- read.csv('~/Downloads/toy_multiclass_1_trainCLEAN.csv', head=T)

toy_test_2 <- read.csv('~/Downloads/toy_multiclass_2_testCLEAN.csv', head=T)
toy_validate_2 <- read.csv('~/Downloads/toy_multiclass_2_validateCLEAN.csv', head=T)
toy_train_2 <- read.csv('~/Downloads/toy_multiclass_2_trainCLEAN.csv', head=T)

toy_1_complete <- rbind(toy_validate_1 %>%
  mutate(type = 'Validation'),
  toy_test_1 %>%
    mutate(type = 'Test'),
  toy_train_1 %>%
  mutate(type = 'Training')
  ) %>%
  mutate(Dataset = as.factor(type),
         `Class` = as.factor(toy.data.3))

toy_2_complete <- rbind(toy_validate_2 %>%
                          mutate(type = 'Validation'),
                        toy_test_2 %>%
                          mutate(type = 'Test'),
                        toy_train_2 %>%
                          mutate(type = 'Training')
) %>%
  mutate(Dataset = as.factor(type),
         `Class` = as.factor(toy.data.3))

t1 <- ggplot(toy_1_complete, aes(x=as.numeric(toy.data.1), y=as.numeric(toy.data.2), 
                       color=Class, shape=Dataset)) + geom_point() + xlab('X1') + ylab('X2') + 
  ggtitle('Distribution of Toy Data Set 1 in 2-dimensions')
ggsave(t1, file='../tex/t1dist.jpg')

t2 <- ggplot(toy_2_complete, aes(x=as.numeric(toy.data.1), y=as.numeric(toy.data.2), 
                           color=Class, shape=Dataset)) + geom_point() + xlab('X1') + ylab('X2') +
  ggtitle('Distribution of Toy Data Set 2 in 2-dimensions')
ggsave(t2, file='t2dist.jpg')

