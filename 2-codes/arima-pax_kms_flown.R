library(seastests)
library(imputeTS)
library(tseries)

pax = read.csv("../data/airlines-PaxKmsFlown.csv")

pax_ts = ts(pax$pax_km, frequency = 12, start = 1988)
plot(pax_ts)

isSeasonal(pax_ts)

pax_ts <- na_interpolation(pax_ts, option = "spline")

pax_ts_diff2 <- diff(pax_ts, differences = 2)

adf.test(pax_ts_diff2)
kpss.test(pax_ts_diff2)

isSeasonal(pax_ts_diff2)

acf(pax_ts_diff2)
pacf(pax_ts_diff2)

pax_arm <- Arima(pax_ts, order = c(6,2,1))
pax_arm <- auto.arima(pax_ts)
plot(forecast(pax_arm, 50, level = 90),
     xlab = "Year",
     ylab = "Kilometres Flown",
     col = "#000000",
     fcol='#d31b27', shadecols="#ffe5e6")
