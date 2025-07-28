dom_ac_mov = read.csv("../data/airports-dom_ac_mov_cleaned.csv")
dam = (dom_ac_mov[,36])

dam = replace(dam, is.na(dam), 1)

adf.test(dam)
kpss.test(dam)

acf(dam)
pacf(dam)

dam_1 = diff(dam, differences = 1)

adf.test(dam_1)
kpss.test(dam_1)

dam_ts = ts(dam)
dam_ts_1 = ts(dam_1, frequency = 1)


acf(dam_ts_1)
pacf(dam_ts_1)

dam_arm = arima(dam_1, order = c(2, 1, 1))

dam_hw = HoltWinters(dam_ts, gamma = F)

plot(dam_ts, col = "blue", xlim = c(0, 150))
lines(dam_hw$fitted[,1])

plot(y=as.numeric(forecast(dam_arm, 10)), x=seq(101, 110))

plot(dam_arm, n.ahead=50)
