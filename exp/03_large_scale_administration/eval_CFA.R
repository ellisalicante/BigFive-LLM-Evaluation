
### 0) IMPORTS
packages <- c(
  "tidyverse",
  "psych",
  "lavaan",
  "semTools",
  "GPArotation",
  "gt",
  "flextable",
  "officer",
  "corrplot",
  "semPlot"
)

library(tidyverse)
library(psych)
library(lavaan)
library(semTools)
library(GPArotation)
library(gt)
library(flextable)
library(officer)
library(corrplot)
library(semPlot)
library(dplyr)
library(tidyr)
library(ggplot2)
library(knitr)



### 1) READING DATA
df <- read.csv("../../dat/03_large_scale_administration/final_dfs/df_cfa.csv", stringsAsFactors = FALSE)

rownames(df) <- df$model
df$model <- NULL
df[] <- lapply(df, as.numeric)

# rename columns safely
colnames(df) <- paste0("x", 1:ncol(df))

str(df)

facets <- list(
  Extraversion = c("x1", "x6", "x11", "x16", "x21", "x26", "x31", "x36"),
  Agreeableness = c("x2", "x7", "x12", "x17", "x22", "x27", "x32", "x37", "x42"),
  Conscientiousness = c("x3", "x8", "x13", "x18", "x23", "x28", "x33", "x38", "x43"),
  Neuroticism = c("x4", "x9", "x14", "x19", "x24", "x29", "x34", "x39"),
  Openness = c("x5", "x10", "x15", "x20", "x25", "x30", "x35", "x40", "x41", "x44")
)



### 2) INTERNAL CONSISTENCY
reliability_results <- lapply(names(facets), function(f) {

  items <- facets[[f]]

  sub <- df[, as.character(items)]

  # Cronbach's alpha
  a <- psych::alpha(sub)

  # McDonald's omega
  o <- psych::omega(sub, plot = FALSE)

  # Guttman's lambda-6
  l6 <- psych::guttman(sub)

  data.frame(
    facet  = f,
    alpha  = a$total$raw_alpha,
    omega  = o$omega.tot,
    lambda = l6$lambda.6
  )
})

reliability_results <- bind_rows(reliability_results)

# round only numeric columns
reliability_results %>%
  mutate(
    across(where(is.numeric), ~ round(.x, 2))
  ) %>%
  print()




### 3) CFA
colnames(df)

model_bfi <- '
Extraversion =~ x1 + x6 + x11 + x16 + x21 + x26 + x31 + x36
Agreeableness =~ x2 + x7 + x12 + x17 + x22 + x27 + x32 + x37 + x42
Conscientiousness =~ x3 + x8 + x13 + x18 + x23 + x28 + x33 + x38 + x43
Neuroticism =~ x4 + x9 + x14 + x19 + x24 + x29 + x34 + x39
Openness =~ x5 + x10 + x15 + x20 + x25 + x30 + x35 + x40 + x41 + x44
'

fit <- cfa(
  model_bfi,
  data = df,
  estimator = "MLR",
  std.lv = TRUE,
)

summary(fit, fit.measures = TRUE, standardized = TRUE)



fa.parallel(
  df,
  fa = "fa",
  fm = "minres",
  n.iter = 100
)



### 4) PLOTTING

impact_six <- c("#008CBB", "#9569D1", "#A3D900", "#E30053", "#FFB000", "#3092FF")

fa_res <- fa.parallel(
  df,
  fa = "fa",
  fm = "minres",
  n.iter = 100,
  plot = FALSE
)

x <- seq_along(fa_res$fa.values)

pdf("fig_scree_plot.pdf", width = 7, height = 5, family = "Times")

par(
  family = "Times",
  cex = 1.2,
  cex.main = 1.2,
  cex.lab = 1.2,
  cex.axis = 1.2
)

# BASE PLOT (actual data)
# triangles + semi-transparent line (alpha 0.9)
plot(
  x, fa_res$fa.values,
  type = "b",
  pch = 17,  # triangles
  lwd = 2,
  col = adjustcolor(impact_six[1], alpha.f = 1),
  xlab = "Factor number",
  ylab = "Eigenvalues of principal factors",
  main = "Parallel Analysis Scree Plot"
)

# SIMULATED DATA (filled dots + dashed line)
lines(
  x, fa_res$fa.sim,
  type = "b",
  pch = 16,  # FILLED circles
  cex = 0.7,
  lty = 2,
  lwd = 1.5,
  col = adjustcolor(impact_six[3], alpha.f = 1)
)

abline(h = 1, lty = 5, col = "grey60")

# LEGEND
legend(
  "topright",
  legend = c("Actual data", "Simulated data"),
  col = c(adjustcolor(impact_six[1], alpha.f = 0.9),
          adjustcolor(impact_six[3], alpha.f = 1)),
  lty = c(1, 2),
  pch = c(17, 16),
  pt.cex = c(1, 0.9),
  cex = 1.2,
  bty = "n"
)

dev.off()



### 5) EFA

efa_1 <- fa(df, nfactors = 1, rotate = "oblimin", fm = "minres")
efa_2 <- fa(df, nfactors = 2, rotate = "oblimin", fm = "minres")
efa_3 <- fa(df, nfactors = 3, rotate = "oblimin", fm = "minres")
efa_4 <- fa(df, nfactors = 4, rotate = "oblimin", fm = "minres")
efa_5 <- fa(df, nfactors = 5, rotate = "oblimin", fm = "minres")

anova(efa_1, efa_2, efa_3, efa_4, efa_5)

summary(efa_2, fit.measures = TRUE, standardized = TRUE)

print(efa_2$loadings, cutoff = 0.3)
