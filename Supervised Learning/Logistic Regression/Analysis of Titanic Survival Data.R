library('msme')
library('rjags')
data(titanic)
barplot(prop.table(table(titanic$survived)), main = 'Density of Passengers Who Survived/Died', xlab = 'Survived? (0 = Died, 1 = Survived)', 
        yaxp = c(0, 0.625, 25), las = 1)
barplot(prop.table(table(titanic$survived, titanic$class)), beside = TRUE, 
        main = 'Density of Passengers Who Survived/Died In Each Class', xlab = 'Class', yaxp = c(0, 0.42, 21), las = 1,
        legend = TRUE, args.legend = list(title = 'Survived?', cex = 1.05, x = 7.2, y = 0.42), space = c(0, 0.4))
barplot(prop.table(table(titanic$survived, titanic$age)), beside = TRUE, 
        main = 'Density of Passengers Who Survived/Died In Each Age Group', yaxp = c(0, 0.6, 30), las = 1, 
        xlab = 'Age Group (0 = Child, 1 = Adult)', legend = TRUE, args.legend = list(title = 'Survived?', cex = 1.05, x = 4.8, y = 0.6), 
        space = c(0, 0.4))
barplot(prop.table(table(titanic$survived, titanic$sex)), beside = TRUE, main = 'Density of Passengers Who Survived/Died For Each Sex', 
        yaxp = c(0, 0.54, 18), las = 1, xlab = 'Sex (0 = Female, 1 = Male)', legend = TRUE, 
        args.legend = list(title = 'Survived?', cex = 1.05, x = 4.8, y = 0.54), space = c(0, 0.4))

model2_string = "model 
{
  for(i in 1:length(Y))
  {
    Y[i] ~ dbern(phi[i])
    logit(phi[i]) = a[class[i]] + b[1]*age[i] + b[2]*sex[i]
  }
  
  for(j in 1:max(class))
  {
    a[j] ~ dnorm(0, 1/(1e6))
  }
  
  for(k in 1:2)
  {
    b[k] ~ dnorm(0, 1/(1e4))
  }
}
"

set.seed(100)
data_jags = list(Y = titanic$survived, class = titanic$class, age = titanic$age, sex = titanic$sex)
params = c('a', 'b')
model2 = jags.model(textConnection(model2_string), data = data_jags, n.chains = 3)
update(model2, 1500)
model2_simulation = coda.samples(model = model2, variable.names = params, n.iter = 10000)
summary(model2_simulation)
model2_simulation_combined = as.mcmc(do.call(rbind, model2_simulation))

X = as.matrix(titanic[, -1])
pm_coefficients2 = colMeans(model2_simulation_combined)
logit_survived_hat2 = pm_coefficients2[c(sprintf('a[%s]', X[, 3]))] + 
  (X[, 1:2] %*% pm_coefficients2[c('b[1]', 'b[2]')])
survived_hat2 = 1/(1 + exp(-logit_survived_hat2))
residual2 = titanic$survived - survived_hat2
plot(residual2)
mean(residual2)
mean(residual2^2)

table0.4 = table(survived_hat2 > 0.4, data_jags$Y)
sum(diag(table0.4))/sum(table0.4)
table0.5 = table(survived_hat2 > 0.5, data_jags$Y)
sum(diag(table0.5))/sum(table0.5)
table0.6 = table(survived_hat2 > 0.6, data_jags$Y)
sum(diag(table0.6))/sum(table0.6)
