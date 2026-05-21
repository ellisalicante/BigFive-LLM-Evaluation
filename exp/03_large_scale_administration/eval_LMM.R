

### 0) IMPORTS
# install.packages("purrr")
# install.packages("broom.mixed")
library(broom.mixed)
library(lme4)
library(lmerTest)
library(emmeans)
library(tidyverse)



### 1) READING DATA
df <- read.csv("../../dat/03_large_scale_administration/final_dfs/df_lmm.csv")

df$model_id  <- as.factor(df$model_id)
df$item_id   <- as.factor(df$item_id)
df$SizeGroup <- as.factor(df$SizeGroup)
df$dimension <- as.factor(df$dimension)

head(df)
str(df)


### 2) LMM (overall)
model <- lmer(
  y ~ SizeGroup +
      ReleaseDate +
      Reasoning +
      OpenWeight +
      (1 | model_id) +
      (1 | item_id) +
      (1 | model_id:item_id),
  data = df,
  REML = FALSE
)
summary(model)



### 3) LLM for OCEAN

split_data <- split(df, df$dimension)

models <- lapply(split_data, function(d) {
  lmer(
    y ~ SizeGroup +
      ReleaseDate +
      Reasoning +
      OpenWeight +
      # y_soc_des +
      (1 | model_id) +
      (1 | item_id) +
      (1 | model_id:item_id),
    data = d,
    REML = FALSE
  )
})

names(models)
summary(models$Agreeableness)
summary(models$Conscientiousness)
summary(models$Extraversion)
summary(models$Neuroticism)
summary(models$Openness)
