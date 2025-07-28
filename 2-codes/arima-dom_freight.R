freight <- read.csv("../data/airports-dom_freight_mov_short.csv", row.names = 1, check.names = F)
freight <- t(freight)
library(imputeTS)
freight <- na_interpolation(freight, option = "spline")
freight <- ts(freight, start = c(2003, 7), frequency = 12)

frs = lapply(freight, function(t) ts(t, start=c(2003, 7), frequency=12))
cities <- names(frs)

for(city in cities){
  name <- 
  png(filename = 
        paste("C:/Users/keval/Documents/Apps/PyCharm/TSA/figs/freight/",
            tolower(city), ".png", sep=""))
  plt <- plot(frs[[city]])
  dev.off()
}

freight_post <- read.csv("../data/airports-dom_freight_mov_post.csv", row.names = 1, check.names = F)
freight_post <- t(freight_post)
freight_post <- na_interpolation(freight_full, option = "spline")
freight_post <- ts(freight_full, start = c(2020, 1), frequency = 12)

frs_post = lapply(freight_post, function(t) ts(t, start=c(2020, 1), frequency=12))

