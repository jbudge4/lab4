# Part 1. Configure your project and examine the data ----

# 1.2. Load the required packages ----
library(glmmTMB)
library(dplyr)
library(tidyr)
library(ggplot2)
library(broom.mixed)
library(performance)
library(gt)
library(readr)
library(DHARMa)
library(ggeffects)
library(titanic)

# 1.3. Load the dataset ----
# Prepare the same complete-case sample for every model
titanic <- titanic_train |>
  select(Survived, Sex, Age, Pclass) |>
  drop_na() |>
  mutate(
    Sex.Binary = case_when(
      Sex == "male" ~ 0,
      Sex == "female" ~ 1
    ),
    Survived.Char = case_when(
      Survived == 0 ~ "Died",
      Survived == 1 ~ "Survived"
    ),
    Pclass.Char = case_when(
      Pclass == 1 ~ "First Class",
      Pclass == 2 ~ "Second Class",
      Pclass == 3 ~ "Third Class"
    ),
    Pclass = factor(Pclass, levels = c(1, 2, 3))
  ) |>
  select(Survived, Survived.Char, Sex, Sex.Binary, Pclass, Pclass.Char, Age)

n <- nrow(titanic)

# 1.4. Explore the data ----
# Inspect the sample in the R Console
print(head(titanic))
str(titanic)

# Summarize age
age_summary <- titanic |>
  summarise(
    Variable = "Age (years)",
    Mean = mean(Age),
    SD = sd(Age),
    Minimum = min(Age),
    Maximum = max(Age)
  )

print(age_summary)

# Count each level of the three categorical variables
survival_counts <- titanic |>
  count(Survived.Char, name = "Count") |>
  transmute(Variable = "Survival", Category = Survived.Char, Count)

sex_counts <- titanic |>
  count(Sex, name = "Count") |>
  transmute(Variable = "Sex", Category = tools::toTitleCase(Sex), Count)

class_counts <- titanic |>
  count(Pclass, Pclass.Char, name = "Count") |>
  transmute(Variable = "Ticket class", Category = Pclass.Char, Count)

categorical_counts <- bind_rows(survival_counts, sex_counts, class_counts)

print(categorical_counts)

# Part 2. Specify competing binary logistic regression models ----

# 2.2. Specify four nested models ----
# Female and first class are the reference categories.
titanic <- titanic |>
  mutate(Sex = factor(Sex, levels = c("female", "male")))

mod0 <- glmmTMB(
  Survived ~ 1,
  data = titanic,
  family = binomial(link = "logit")
)

mod1 <- glmmTMB(
  Survived ~ Sex,
  data = titanic,
  family = binomial(link = "logit")
)

mod2 <- glmmTMB(
  Survived ~ Sex + Pclass,
  data = titanic,
  family = binomial(link = "logit")
)

mod3 <- glmmTMB(
  Survived ~ Sex + Pclass + Age,
  data = titanic,
  family = binomial(link = "logit")
)

# 2.3. Explore the fitted models in the R Console ----
print(summary(mod0))
print(summary(mod1))
print(summary(mod2))
print(summary(mod3))

# Part 3. Compare model performance ----

# 3.1. Conduct nested-model chi-square tests ----
nested_tests <- anova(mod0, mod1, mod2, mod3)

# Df is the number of fitted parameters for each model.
nested_results <- nested_tests |>
  mutate(Residual.df = n - Df) |>
  select(Residual.df, `Chi Df`, Chisq, `Pr(>Chisq)`) |>
  rename(
    Nested.df = `Chi Df`,
    Nested.Chisq = Chisq,
    Nested.p = `Pr(>Chisq)`
  )

print(nested_tests)

# 3.2. Extract model-level performance statistics ----
model_metrics <- bind_rows(
  mod0 = broom.mixed::glance(mod0),
  mod1 = broom.mixed::glance(mod1),
  mod2 = broom.mixed::glance(mod2),
  mod3 = broom.mixed::glance(mod3),
  .id = "Model"
) |>
  mutate(Deviance = -2 * logLik) |>
  select(Model, AIC, BIC, logLik, Deviance)

# 3.3. Create a combined model-comparison table ----
# Both data frames list mod0 through mod3 in the same order.
comparison_table <- bind_cols(model_metrics, nested_results)

print(comparison_table)

# Part 4. Evaluate binary logistic regression diagnostics ----

# 4.1. Examine the assumptions ----

# Residual diagnostics ----
set.seed(9041)
res_mod3 <- DHARMa::simulateResiduals(mod3, n = 500, plot = FALSE)
plot(res_mod3)

# Collinearity ----
print(performance::check_collinearity(mod3))

# Complete separation ----
# Check for unusually large coefficients or standard errors.
print(summary(mod3)$coefficients$cond)

# Check whether fitted probabilities are effectively 0 or 1.
print(range(fitted(mod3)))

# Potential outliers ----
print(DHARMa::testOutliers(res_mod3, plot = FALSE))

# Events per predictor coefficient ----
n_event <- sum(titanic$Survived == 1)
n_nonevent <- sum(titanic$Survived == 0)

# Count the two ticket-class coefficients separately and exclude the intercept.
n_coef <- length(fixef(mod3)$cond) - 1
print(c(Survivors = n_event, Deaths = n_nonevent, Predictor.Coefficients = n_coef))
print(c(Survivors = n_event, Deaths = n_nonevent) / n_coef)

# Part 5. Select and report your preferred model ----

# 5.1. Select a model ----
selected_model <- mod3

# 5.2. Extract coefficients and 95% confidence intervals ----
coefficient_results <- broom.mixed::tidy(
  selected_model,
  effects = "fixed",
  component = "cond",
  conf.int = TRUE,
  conf.level = 0.95
) |>
  mutate(
    odds_ratio = exp(estimate),
    or_low = exp(conf.low),
    or_high = exp(conf.high)
  )

print(as.data.frame(coefficient_results |>
  select(term, estimate, std.error, conf.low, conf.high, p.value, odds_ratio, or_low, or_high)))

# 5.4. Report overall model performance ----
selected_row <- comparison_table |>
  filter(Model == "mod3")

print(as.data.frame(selected_row))

# 5.6. Visualize one predictor ----
pred_age <- ggeffects::ggpredict(
  selected_model,
  terms = "Age [all]",
  condition = c(Sex = "female", Pclass = "1")
)

age_plot <- ggplot(pred_age, aes(x, predicted)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.18) +
  geom_line(linewidth = 1.2) +
  scale_y_continuous(labels = scales::label_percent(), limits = c(0, 1)) +
  labs(
    x = "Age (years)",
    y = "Predicted probability of survival"
  ) +
  theme_classic()

print(age_plot)
