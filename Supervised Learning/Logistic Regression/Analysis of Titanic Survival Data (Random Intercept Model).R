library('msme')
library('rjags')
data(titanic)
barplot(prop.table(table(titanic$survived)), main = 'Density of Passengers Who Survived/Died', xlab = 'Survived? (0 = Died, 1 = Survived)', 
        yaxp = c(0, 0.625, 25), las = 1, cex.axis = 1.1)
barplot(prop.table(table(titanic$survived, titanic$class)), beside = TRUE, 
        main = 'Density of Passengers Who Survived/Died In Each Class', xlab = 'Class', yaxp = c(0, 0.42, 21), las = 1, cex.axis = 1.25,
        legend = TRUE, args.legend = list(title = 'Survived?', cex = 1.05, x = 7.2, y = 0.42), space = c(0, 0.4))
barplot(prop.table(table(titanic$survived, titanic$age)), beside = TRUE, 
        main = 'Density of Passengers Who Survived/Died In Each Age Group', yaxp = c(0, 0.6, 30), las = 1,  cex.axis = 1.15,
        xlab = 'Age Group (0 = Child, 1 = Adult)', legend = TRUE, args.legend = list(title = 'Survived?', cex = 1.05, x = 4.8, y = 0.6), 
        space = c(0, 0.4))
barplot(prop.table(table(titanic$survived, titanic$sex)), beside = TRUE, main = 'Density of Passengers Who Survived/Died For Each Sex', 
        xlab = 'Sex (0 = Female, 1 = Male)', yaxp = c(0, 0.54, 18), las = 1, cex.axis = 1.1 ,legend = TRUE, 
        args.legend = list(title = 'Survived?', cex = 1.05, x = 4.8, y = 0.54), space = c(0, 0.4))

model_string = "model 
{
  for(i in 1:length(Y))
  {
    Y[i] ~ dbern(phi[i])
    logit(phi[i]) = a[class[i]] + b[1]*age[i] + b[2]*sex[i]
  }
  
  for(j in 1:max(class))
  {
    a[j] ~ dnorm(mu, tau2)
  }
  mu ~ dnorm(0, 1/(1e6)) 
  tau2 ~ dgamma(1, 25)
  sd = sqrt(1/tau2)
  
  for(k in 1:2)
  {
    b[k] ~ dnorm(0, 1/(1e4))
  }
}
"

set.seed(100)
data_jags = list(Y = titanic$survived, class = titanic$class, age = titanic$age, sex = titanic$sex)
params = c('a', 'b', 'mu', 'sd')
model = jags.model(textConnection(model_string), data = data_jags, n.chains = 3)
update(model, 1500)
model_simulation = coda.samples(model = model, variable.names = params, n.iter = 10000)
summary(model_simulation)
model_simulation_combined = as.mcmc(do.call(rbind, model_simulation))

X = as.matrix(titanic[-1])
pm_coefficients = colMeans(model_simulation_combined)
logit_survived_hat = pm_coefficients[c(sprintf('a[%s]', X[, 3]))] + (X[, 1:2] %*% pm_coefficients[c('b[1]', 'b[2]')])
survived_hat = 1/(1 + exp(-logit_survived_hat))
residual = titanic$survived - survived_hat
plot(residual)
mean(residual)
mean(residual^2)

table0.4 = table(survived_hat > 0.4, data_jags$Y)
sum(diag(table0.4))/sum(table0.4)
table0.5 = table(survived_hat > 0.5, data_jags$Y)
sum(diag(table0.5))/sum(table0.5)
# Ideal for risk evaluation (Better to predict more deaths (FALSE) than actual deaths [0])
table0.6 = table(survived_hat > 0.6, data_jags$Y)
sum(diag(table0.6))/sum(table0.6)

# Girl (Sex = 0, Age = 0)
girl_first_class = c(0, 0, 1)
logit_girl1_survival_prediction = model_simulation_combined[, c(sprintf('a[%s]', girl_first_class[3]))] + 
  (model_simulation_combined[, c('b[1]', 'b[2]')] %*% girl_first_class[1:2])
girl1_survival_prediction = 1/(1+exp(-logit_girl1_survival_prediction))
mean(girl1_survival_prediction)
girl_second_class = c(0, 0, 2)
logit_girl2_survival_prediction = model_simulation_combined[, c(sprintf('a[%s]', girl_second_class[3]))] + 
  (model_simulation_combined[, c('b[1]', 'b[2]')] %*% girl_second_class[1:2])
girl2_survival_prediction = 1/(1+exp(-logit_girl2_survival_prediction))
mean(girl2_survival_prediction)
girl_third_class = c(0, 0, 3)
logit_girl3_survival_prediction = model_simulation_combined[, c(sprintf('a[%s]', girl_third_class[3]))] + 
  (model_simulation_combined[, c('b[1]', 'b[2]')] %*% girl_third_class[1:2])
girl3_survival_prediction = 1/(1+exp(-logit_girl3_survival_prediction))
mean(girl3_survival_prediction)

# Boy (Sex = 1, Age = 0)
boy_first_class = c(1, 0, 1)
logit_boy1_survival_prediction = model_simulation_combined[, c(sprintf('a[%s]', boy_first_class[3]))] + 
  (model_simulation_combined[, c('b[1]', 'b[2]')] %*% boy_first_class[1:2])
boy1_survival_prediction = 1/(1+exp(-logit_boy1_survival_prediction))
mean(boy1_survival_prediction)
boy_second_class = c(1, 0, 2)
logit_boy2_survival_prediction = model_simulation_combined[, c(sprintf('a[%s]', boy_second_class[3]))] + 
  (model_simulation_combined[, c('b[1]', 'b[2]')] %*% boy_second_class[1:2])
boy2_survival_prediction = 1/(1+exp(-logit_boy2_survival_prediction))
mean(boy2_survival_prediction)
boy_third_class = c(1, 0, 3)
logit_boy3_survival_prediction = model_simulation_combined[, c(sprintf('a[%s]', boy_third_class[3]))] + 
  (model_simulation_combined[, c('b[1]', 'b[2]')] %*% boy_third_class[1:2])
boy3_survival_prediction = 1/(1+exp(-logit_boy3_survival_prediction))
mean(boy3_survival_prediction)

# Woman (Sex = 0, Age = 1)
woman_first_class = c(0, 1, 1)
logit_woman1_survival_prediction = model_simulation_combined[, c(sprintf('a[%s]', woman_first_class[3]))] + 
  (model_simulation_combined[, c('b[1]', 'b[2]')] %*% woman_first_class[1:2])
woman1_survival_prediction = 1/(1+exp(-logit_woman1_survival_prediction))
mean(woman1_survival_prediction)
woman_second_class = c(0, 1, 2)
logit_woman2_survival_prediction = model_simulation_combined[, c(sprintf('a[%s]', woman_second_class[3]))] + 
  (model_simulation_combined[, c('b[1]', 'b[2]')] %*% woman_second_class[1:2])
woman2_survival_prediction = 1/(1+exp(-logit_woman2_survival_prediction))
mean(woman2_survival_prediction)
woman_third_class = c(0, 1, 3)
logit_woman3_survival_prediction = model_simulation_combined[, c(sprintf('a[%s]', woman_third_class[3]))] + 
  (model_simulation_combined[, c('b[1]', 'b[2]')] %*% woman_third_class[1:2])
woman3_survival_prediction = 1/(1+exp(-logit_woman3_survival_prediction))
mean(woman3_survival_prediction)

# Man (Sex = 1, Age = 1)
man_first_class = c(1, 1, 1)
logit_man1_survival_prediction = model_simulation_combined[, c(sprintf('a[%s]', man_first_class[3]))] + 
  (model_simulation_combined[, c('b[1]', 'b[2]')] %*% man_first_class[1:2])
man1_survival_prediction = 1/(1+exp(-logit_man1_survival_prediction))
mean(man1_survival_prediction)
man_second_class = c(1, 1, 2)
logit_man2_survival_prediction = model_simulation_combined[, c(sprintf('a[%s]', man_second_class[3]))] + 
  (model_simulation_combined[, c('b[1]', 'b[2]')] %*% man_second_class[1:2])
man2_survival_prediction = 1/(1+exp(-logit_man2_survival_prediction))
mean(man2_survival_prediction)
man_third_class = c(1, 1, 3)
logit_man3_survival_prediction = model_simulation_combined[, c(sprintf('a[%s]', man_third_class[3]))] + 
  (model_simulation_combined[, c('b[1]', 'b[2]')] %*% man_third_class[1:2])
man3_survival_prediction = 1/(1+exp(-logit_man3_survival_prediction))
mean(man3_survival_prediction)
