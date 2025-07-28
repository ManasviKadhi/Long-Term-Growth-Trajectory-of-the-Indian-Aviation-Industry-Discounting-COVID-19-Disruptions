kol_f <- frs$Kolkata
del_f <- frs$Delhi
mum_f <- frs$Mumbai
ben_f <- frs$Bangalore
pun_f <- frs$Pune
che_f <- frs$Chennai

kol_fp <- frs_post$Kolkata
del_fp <- frs_post$Delhi
mum_fp <- frs_post$Mumbai
ben_fp <- frs_post$Bangalore
pun_fp <- frs_post$Pune
che_fp <- frs_post$Chennai

dsize <- seq(2003, 2019, length.out = 198)
fsize <- seq(2020, 2024, length.out = 56)

cmum = "#ae81c9" # (lavender) - mumbai
ckol = "#575494" # (indigo) - kolkata
cben = "#bababa" # (grey) - bangalore
cpun = "#7e7cc5" # (dark lavender) - pune
cdel = "#576e9a" # (blue-grey) - delhi
cche = "#7aa6b2" # (light blue-grey) - chennai

kol_a <- auto.arima(kol_f)
del_a <- auto.arima(del_f)
mum_a <- auto.arima(mum_f)
ben_a <- auto.arima(ben_f)
pun_a <- auto.arima(pun_f)
che_a <- auto.arima(che_f)

# plot(forecast(kol_a, 50, level = 90), ylim = c(0, 13000), col = ckol, fcol = "black", lwd = 2, xlab = "Year", ylab = "Freight Movement")

fkol <- forecast(kol_a, 56, level = 90)
fdel <- forecast(del_a, 56, level = 90)
fmum <- forecast(mum_a, 56, level = 90)
fben <- forecast(ben_a, 56, level = 90)
fpun <- forecast(pun_a, 56, level = 90)
fche <- forecast(che_a, 56, level = 90)


plot(dsize, fkol$x, xlim = c(2003, 2024), ylim = c(0, 40000), type = "l", col = ckol, lwd = 2, xlab = "Year", ylab = "Freight Movement")
lines(fsize, kol_fp, col = ckol, lwd = 2)
lines(fsize, fkol$mean, col = ckol, lwd = 3)

lines(dsize, fdel$x, col = cdel, lwd = 2)
plot(dsize, fdel$x, xlim = c(2003, 2024), ylim = c(0, 40000), type = "l", col = cben, lwd = 2, xlab = "Year", ylab = "Freight Movement")
lines(fsize, del_fp, col = cben, lwd = 2)
lines(fsize, fdel$mean, col = cben, lwd = 3)

lines(dsize, fben$x, col = cben, lwd = 2)
lines(fsize, ben_fp, col = cben, lwd = 2)
lines(fsize, fben$mean, col = cben, lwd = 3)

lines(dsize, fmum$x, col = cmum, lwd = 2)
lines(fsize, mum_fp, col = cmum, lwd = 2)
lines(fsize, fmum$mean, col = cmum, lwd = 3)

lines(dsize, fpun$x, col = cpun, lwd = 2)
lines(fsize, pun_fp, col = cpun, lwd = 2)
lines(fsize, fpun$mean, col = cpun, lwd = 3)

lines(dsize, fche$x, col = cche, lwd = 2)
lines(fsize, che_fp, col = cche, lwd = 2)
lines(fsize, fche$mean, col = cche, lwd = 3)
