# generate simulation data

set.seed(1);

# number of variables
p <- 100;
# number of samples
n <- 20;

# low-level shift
delta1 <- 0.2;
# high-level shift
delta2 <- 1;

# data matrix
x <- matrix(rnorm(p * n, sd=0.3), nrow=p, ncol=n);

# broad loss
x[31:90, 1:15] <- x[31:90, 1:15] - delta1;

# focal amplification
x[65:70, 5:10] <- x[65:70, 5:10] + delta2;
# focal deletion
x[40:45, 10:15] <- x[40:45, 10:15] - delta2;

# focal amplification in all samples
x[90:95 ,] <- x[90:95 ,] + delta2
# focal deletion in all samples
x[10:15 ,] <- x[10:15 ,] - delta2


dot <- function(x, y) {
	sum(x * y)
}

cosine_similarity <- function(x, y) {
	dot(x, y) / ( sqrt(dot(x, x)) * sqrt(dot(y, y)) )
}

scores <- numeric(p);
wsize <- 5;
for (i in 1:p) {
	start <- i - floor(wsize/2);
	if (start < 1) start <- 1;
	end <- i + floor(wsize/2);
	if (end > p) end <- p;
	
	# sum cosine similarities within window
	s <- 0;
	for (ii in start:end) {
		s <- s + cosine_similarity(x[i, ], x[ii, ]);
		#s <- s + cor(x[i, ], x[ii, ]);
	}    
	
	scores[i] <- s / (end - start + 1);
}


# diagnostic plots

par(mfrow=c(3, 1));

colours <- colorRampPalette(c("blue", "white", "red"))(100);
image(x, col=colours, xlab="position", ylab="sample");

plot(scores, type="l", ylab="score", xlab="position");

hist(scores, breaks=25, xlab="score");


# write output

write.table(matrix(c("#", dim(x)), ncol=3), "input.tsv", sep="\t", row.names=FALSE, col.names=FALSE, quote=FALSE);
write.table(x, "input.tsv", sep="\t", row.names=FALSE, col.names=FALSE, quote=FALSE, append=TRUE);
write.table(scores, "raw-scores_r.vtr", row.names=FALSE, col.names=FALSE, quote=FALSE);

