library(forecast)

dam = read.csv("../data/airports-dom_ac_mov_short.csv")
dam_post = read.csv("../data/airports-dom_ac_mov_post.csv")

kol <- dam$west_bengal_kolkata_international
del <- dam$delhi_delhi_joint_venture_international
mum <- dam$maharashtra_mumbai_joint_venture_international
ben <- dam$karnataka_bengaluru_joint_venture_international
pun <- dam$maharashtra_pune_customs
che <- dam$tamil_nadu_chennai_international

kol_p <- dam_post$west_bengal_kolkata_international
del_p <- dam_post$delhi_delhi_joint_venture_international
mum_p <- dam_post$maharashtra_mumbai_joint_venture_international
ben_p <- dam_post$karnataka_bengaluru_joint_venture_international
pun_p <- dam_post$maharashtra_pune_customs
che_p <- dam_post$tamil_nadu_chennai_international

size1 <- seq(1, 57)
size2 <- seq(58, 101)
size <- seq(1, 101, length.out = 10)

linreg1 = lm(kol~dam$index)
c1 <- linreg1$coefficients[1]
m1 <- linreg1$coefficients[2]

linreg2 = lm(del~dam$index)
c2 <- linreg2$coefficients[1]
m2 <- linreg2$coefficients[2]

linreg3 = lm(mum~dam$index)
c3 <- linreg3$coefficients[1]
m3 <- linreg3$coefficients[2]

linreg4 = lm(ben~dam$index)
c4 <- linreg4$coefficients[1]
m4 <- linreg4$coefficients[2]

linreg5 = lm(pun~dam$index)
c5 <- linreg5$coefficients[1]
m5 <- linreg5$coefficients[2]

linreg6 = lm(che~dam$index)
c6 <- linreg6$coefficients[1]
m6 <- linreg6$coefficients[2]

preds1 <- c()
preds2 <- c()
preds3 <- c()
preds4 <- c()
preds5 <- c()
preds6 <- c()

for(i in 1:101){
  preds1[i] <- m1*i + c1
  preds2[i] <- m2*i + c2
  preds3[i] <- m3*i + c3
  preds4[i] <- m4*i + c4
  preds5[i] <- m5*i + c5
  preds6[i] <- m6*i + c6
}

plot(kol, col="#575494",
     type = 'l',
     lwd = 3,
     xlim=c(0, 101),
     ylim = c(0, 38000),
     xlab = "Year",
     ylab = "Aircraft Movement",
     xaxt = "n")
axis(1, at = size, labels = c(2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024))
lines(del, col="#576e9a", lwd=3)
lines(mum, col="#ae81c9", lwd=3)
lines(ben, col="#bababa", lwd=3)
lines(pun, col="#7e7cc5", lwd=3)
lines(che, col="#7aa6b2", lwd=3)


lines(preds1, col="#575494", lwd = 3)
lines(preds2, col="#576e9a", lwd = 3)
lines(preds3, col="#ae81c9", lwd = 3)
lines(preds4, col="#bababa", lwd = 3)
lines(preds5, col="#7e7cc5", lwd = 3)
lines(preds6, col="#7aa6b2", lwd = 3)

lines(size2, kol_p, col="#575494", lwd = 2.5)
lines(size2, del_p, col="#576e9a", lwd = 2.5)
lines(size2, mum_p, col="#ae81c9", lwd = 2.5)
lines(size2, ben_p, col="#bababa", lwd = 2.5)
lines(size2, pun_p, col="#7e7cc5", lwd = 2.5)
lines(size2, che_p, col="#7aa6b2", lwd = 2.5)

