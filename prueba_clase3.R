rm(list = rm())

library(ggplot2)

ggplot(penguins_df) +
 aes(x = bill_length_mm, fill = bill_depth_mm) +
 geom_histogram(bins = 30L) +
 scale_fill_gradient() +
 theme_minimal()

ggplot(penguins_df) +
 aes(x = bill_length_mm, fill = bill_depth_mm) +
 geom_histogram(bins = 30L) +
 scale_fill_gradient() +
 theme_minimal()
library(tidyverse)
library(palmerpenguins)
penguins

penguins_df <- penguins %>%
  filter(!is.na(sex)) %>%
  select(-island)

library(tidymodels)
set.seed(123)
penguin_split <- initial_split(penguins_df, strata = sex)
penguin_train <- training(penguin_split)
penguin_test <- testing(penguin_split)
penguin_split

penguin_boot <- bootstraps(penguin_train)
penguin_boot


glm_spec <- logistic_reg() %>%
  set_engine("glm")
glm_spec

rf_spec <- rand_forest() %>%
  set_mode("classification") %>%
  set_e
penguin_wf <- workflow() %>%
  add_formula(sex ~ .)
penguin_wf

glm_rs <- penguin_wf %>%
  add_model(glm_spec) %>%
  fit_resamples(
    resamples = penguin_boot,
    control = control_resamples(save_pred = TRUE)
  )
glm_rs

rf_rs <- penguin_wf %>%
  add_model(rf_spec) %>%
  fit_resamples(
    resamples = penguin_boot,
    control = control_resamples(save_pred = TRUE)
  )
rf_rs

collect_metrics(glm_rs)
collect_metrics(rf_rs)

glm_rs %>%
  conf_mat_resampled()

penguin_final <- penguin_wf %>%
  add_model(glm_spec) %>%
  last_fit(penguin_split)
penguin_final

library(ggplot2)
  
ggplot(penguin_test) +
  aes(
    x = bill_length_mm,
    y = bill_depth_mm,
    colour = sex,
    size = body_mass_g
  ) +
  geom_point(shape = "circle") +
  scale_color_hue(direction = 1) +
  theme_minimal() +
  facet_wrap(vars(species))
