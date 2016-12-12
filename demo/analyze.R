# analyze modes

#modes <- as.matrix(read.table("output/modes.tsv"));
#scores <- read.table("output/raw-scores.vtr")[,1];
#k <- nrow(modes);
#y <- scores - modes[k,];

y <- read.table("output/scores.vtr")[,1];

par(mfrow=c(2,1));
plot(y, type="l");
hist(y);


library(mixtools);

fit <- normalmixEM(y, k=2);
i <- which.min(fit$mu);
mu <- fit$mu[i];
sigma <- fit$sigma[i];

hist(y, freq=FALSE);
curve(dnorm(x, mu, sigma), add=TRUE);

alpha <- 0.05;
# critical value for y
yc <- qnorm(1 - alpha, mu, sigma);

# standardize the detrended scores
z <- (y - mu) / sigma;

# critical value for z
zc <- qnorm(1 - alpha);

par(mfrow=c(2, 1));

plot(y, type="l");
abline(h=yc);

plot(z, type="l");
abline(h=zc);

