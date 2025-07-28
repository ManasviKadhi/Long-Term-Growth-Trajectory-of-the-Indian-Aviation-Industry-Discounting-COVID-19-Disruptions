total <- read.csv("../data/airports-dom_freight_mov-totaled.csv", check.names = F)$Total
pre <- total[1:198]
post <- total[199:254]

library(forecast, tseries)

size <- seq(6, 254, length.out = 21)
years <- seq(2004, 2024)

plot(pre, xlim = c(0, 254), ylim = c(0, max(pre, post)+7000),
     xlab = "Year", ylab = "Freight Movement",
     type = 'l', xaxt = "n")
axis(1, at = size, labels = years)
lr <- lm(pre~seq(1, 198))
abline(a=lr$coefficients[1], b=lr$coefficients[2])
lines(seq(199, 254), post, col = "#d31b27")

total_ts <- ts(total, frequency = 1)
pre_ts <- ts(total_ts[1:198], frequency = 1)
post_ts <- ts(total_ts[199:254], frequency = 1)
total_tslm <- tslm(pre_ts~seq(1,198))

frarm <- auto.arima(pre)

plot(forecast(frarm, 56, level = 90), xaxt = "n",
     xlim = c(0, 254), ylim = c(0, max(forecast(frarm, 56, level = 90)$upper)+7000),
     xlab = "Year",
     ylab = "Freight Movement",
     col = "#000000",
     fcol='#d31b27', shadecols="#ffe5e6")
axis(1, at = size, labels = years)

lines(seq(199, 254), post, col = "black")
