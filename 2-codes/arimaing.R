library(forecast)
library(tseries)
temp <- frs$Kolkata
temp_post <- frs_post$Kolkata

plot(temp, xlim = c(0, ))
adf.test(temp)
kpss.test(temp)

temp1 <- diff(temp, differences = 1)
plot(temp1)
adf.test(temp1)
kpss.test(temp1) # passed? d = 1

acf(temp1) # gives q
pacf(temp1) # gives p

temp_arm <- Arima(temp, order = c(1, 1, 1))
plot(
  forecast(temp_arm, 60, level = 90),
  xlab = "Time",
  ylab = "Freight Movement",
  col = "#000000",
  fcol='#ff0000', shadecols="#f6dadb",
  )

lines(temp_full)
