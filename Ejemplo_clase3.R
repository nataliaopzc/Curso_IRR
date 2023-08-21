# Paquetes ----

library(tidytuesdayR)
library(skimr)
library(themis)
library(tidyverse)
library(tidymodels)

# Datos ----
tt_data <- tidytuesdayR::tt_load(2020,week=39)
tt_data$members %>% # Análisis exploratorio
skimr::skim() # más sofisticado que el glimpse

## Climbers DF----
# Selecciono los datos que esán a continuación del set de datos membrs
climbers_df <- tt_data$members |> 
  select(member_id, peak_name, season, year, sex, age, 
         citizenship,expedition_role, hired, solo, oxygen_used,
         success, died) |> 

# Filtro todos los nan que están en las variable indicadas
  filter((!is.na(sex) & !is.na(citizenship) & !is.na(peak_name) &
            !is.na(expedition_role)) == T) |> 
# Transformar los caracteres y lógicos a factor
  mutate(across(where(~ is.character(.) | is.logical(.)),
                as.factor))

# Data split ----
## Semilla ----
set.seed(2023)

## Split inicial ----
#De la variable seleccionada voy a dividirlaen dos, donde el 80% 
#son de testeo y el resto de entrenamiento
climbers_split <- initial_split(climbers_df, prop = 0.8, strata = died)
climbers_split

## Conjunto de entrenamiento ----
train_set <- training(climbers_split)
  
## Conjunto de prueba ----
test_set <- testing(climbers_split)

## CV ----
climbers_fold <- train_set |> 
  vfold_cv(v = 10, repeats = 1, strata = died)

# Recetas ----
mod_recipe <- recipe(formula = died ~ ., data = train_set)


# Los datos que en la variable age son faltante
# voy a incorporar la mediana, luego normalizo todos los 
# predictores que son numéricos (resto media y div, por sv)
mod_recipe <- mod_recipe |>
  update_role(member_id, new_role = "id") |> 
  step_impute_median(age) |> 
  step_normalize(all_numeric_predictors()) |>
  step_other(peak_name, citizenship, expedition_role, threshold = 0.05) |> 
  step_dummy(all_predictors(), -all_numeric(), one_hot = F) |> 
  step_upsample(died, over_ratio = 0.2, seed = 2023, skip = T)

## Preparación reseta ----
mod_recipe_prepped <- prep(mod_recipe, retain = T)

## Bakr ----
bake(mod_recipe_prepped, new_data = NULL)

# Modelos ----
## Regresión logística glm ----
log_cls <- logistic_reg() |> 
  set_engine('glm') |> 
  set_mode("classification")
log_cls

## Regresión logística glmnet ----
reg_log_cls <- logistic_reg() |>
  set_args(penaly = tune(), mixture = tune()) |>  set_mode("classification") |> 
  set_engine("glmnet", family ="binomial")
reg_log_cls

## Workflow ----

cls_wf <- workflow() |> 
  add_recipe(mod_recipe) |> 
  add_model(reg_log_cls)
cls_wf
