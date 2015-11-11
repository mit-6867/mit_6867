library(ggplot2)
library(dplyr)
library(scales)

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
  ggtitle('Scatterplot of Toy Data Set 1 in 2-dimensions')
ggsave(t1, file='../tex/t1dist.jpg')

t2 <- ggplot(toy_2_complete, aes(x=as.numeric(toy.data.1), y=as.numeric(toy.data.2), 
                           color=Class, shape=Dataset)) + geom_point() + xlab('X1') + ylab('X2') +
  ggtitle('Scatterplot of Toy Data Set 2 in 2-dimensions')
ggsave(t2, file='../tex/t2dist.jpg')

nn_performance <- read.csv('nnperformance.csv', head=T)
head(nn_performance)

nnperf <- nn_performance %>% 
  mutate(`Learning Rate` = as.factor(lr),
         `Regularization` = as.factor(lambda)) %>%
  ggplot(aes(x=n, y=error, color=`Regularization`)) + 
  geom_line(aes(linetype=`Learning Rate`), size=1) + scale_y_continuous(labels=percent) + facet_wrap(~method) + 
  xlab('Hidden Units') + ylab('Classification Error Rate') + 
  ggtitle('Neural Network Classification Error on MNIST Validation Dataset')
ggsave(nnperf, file='../tex/nnperf.jpg')