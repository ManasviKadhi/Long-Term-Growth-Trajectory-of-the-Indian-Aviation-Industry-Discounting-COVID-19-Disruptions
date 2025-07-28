load <- read.csv("../data/airlines-PaxLoad.csv")
carried <- read.csv("../data/airlines-PaxCarried.csv")

pax_load <- as.numeric(load$pax_load)
pax_carried <- as.numeric(carried$pax_carried)

load_ts <- ts(pax_load, frequency = 12, start = c(1988, 5))
carried_ts <- ts(pax_carried, frequency = 12, start = 1988)

load_ts <- na_interpolation(load_ts)
carried_ts <- na_interpolation(carried_ts)

plot(load_ts)
plot(carried_ts)

load_1 <- diff(load_ts, differences = 1)

acf(load_1)
pacf(load_1)

zero_load <- load_ts - mean(load_ts)

load_arm <- auto.arima(load_ts)
plot(forecast(load_arm, 50),
     xlab = "Year",
     ylab = "Passenger Load",
     col = "#000000",
     fcol='#d31b27', shadecols="#ffe5e6")

carried_arm <- auto.arima(carried_ts)
plot(forecast(carried_arm, 50),
     xlab = "Year",
     ylab = "Passengers Carried",
     col = "#000000",
     fcol='#d31b27', shadecols="#ffe5e6")
)

carried_ts1 <- diff(carried_ts, differences = 1)
acf(carried_ts1)
pacf(carried_ts1)

carried_arm1 <- Arima(carried_ts, order = c(1, 1, 1))
plot(forecast(carried_arm1, 50))
