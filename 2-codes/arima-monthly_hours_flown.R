monthly_hours = read.csv("../data/airlines-MonthlyHoursFlown.csv")

library(imputeTS)
library(forecast)
hrs = as.numeric(monthly_hours$hours_flown)


hrs_ts <- ts(hrs, frequency = 12, start = 1988)
hrs_ts <- na_interpolation(hrs_ts)

hrs_lm = lm(hrs~seq(308))
plot(hrs)
abline(reg = hrs_lm)

coeff=coefficients(hrs_lm)


hrs_ts_diff1 <- diff(hrs_ts, differences = 1)

adf.test(hrs_ts_diff1)
kpss.test(hrs_ts_diff1)

isSeasonal(hrs_ts_diff1)

acf(hrs_ts_diff1)
pacf(hrs_ts_diff1)

hrss <- auto.arima(hrs_ts)
plot(forecast(hrss, 50))

hrs_arm <- auto.arima(hrs_ts)
plot(forecast(hrs_arm, 50, level = 90),
     xlab = "Year",
     ylab = "Monthly Hours Flown",
     col = "#000000",
     fcol='#d31b27', shadecols="#ffe5e6")

     