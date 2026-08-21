#!/usr/bin/env Rscript
# Fit the classical Royston-Parmar reference models with rstpm2 and export
# everything the PyTorch equivalence test needs as JSON.
#
# Cause-specific fits: for cause k, competing events are censored
# (Surv(time, event == k)), df = 4 spline on log(time).
#
# Exports per cause:
#   - knots actually used by rstpm2 (log-time scale): boundary + internal
#   - fitted coefficients (nsx basis parameterisation, for the record)
#   - maximised log-likelihood
#   - log cumulative hazard predictions on a dense 200-point time grid for
#     the baseline profile and 5 reference covariate profiles (the
#     parameterisation-robust comparison object)
#
# Run once:  Rscript rp_core/reference/fit_reference.R
# Writes:    rp_core/reference/r_reference.json

suppressPackageStartupMessages({
  library(rstpm2)
  library(survival)
  library(bbmle)   # S4 logLik method for the underlying mle2 objects
  library(jsonlite)
})

loglik_of <- function(fit) {
  ll <- tryCatch(as.numeric(stats4::logLik(fit)), error = function(e) NULL)
  if (is.null(ll)) ll <- -as.numeric(fit@min)  # mle2 stores the minimised NLL
  ll
}

args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(sub("^--file=", "", args[grepl("^--file=", args)]))
if (length(script_dir) == 0 || script_dir == ".") script_dir <- "rp_core/reference"

d <- read.csv(file.path(script_dir, "sim_data.csv"))
stopifnot(all(c("time", "event", "x1", "x2", "x3", "x4") %in% names(d)))

# Dense evaluation grid: 1st-99th percentile of observed times, log-spaced.
tgrid <- exp(seq(log(quantile(d$time, 0.01)), log(quantile(d$time, 0.99)),
                 length.out = 200))

# 5 reference covariate profiles (fixed, mirrored in the Python test).
profiles <- data.frame(
  x1 = c(0, 1, -1, 0.5, 2),
  x2 = c(0, -1, 1, 0.5, -2),
  x3 = c(0, 1, 0, 1, 0),
  x4 = c(0, 0, 1, 1, 0)
)

fit_one <- function(k) {
  dd <- d
  dd$ev <- as.integer(dd$event == k)
  fit <- stpm2(Surv(time, ev) ~ x1 + x2 + x3 + x4, data = dd, df = 4)

  # Extract the exact nsx() knots rstpm2 used (log-time scale).
  mf <- fit@model.frame
  nsx_name <- grep("nsx", names(mf), value = TRUE)
  stopifnot(length(nsx_name) == 1)
  basis <- mf[[nsx_name]]
  internal_knots <- as.numeric(attr(basis, "knots"))
  boundary_knots <- as.numeric(attr(basis, "Boundary.knots"))

  # log H predictions: baseline profile x = 0 and the 5 reference profiles.
  pred_profiles <- lapply(seq_len(nrow(profiles)), function(i) {
    nd <- profiles[rep(i, length(tgrid)), , drop = FALSE]
    nd$time <- tgrid
    as.numeric(log(predict(fit, newdata = nd, type = "cumhaz")))
  })
  nd0 <- data.frame(x1 = 0, x2 = 0, x3 = 0, x4 = 0, time = tgrid)
  pred_baseline <- as.numeric(log(predict(fit, newdata = nd0, type = "cumhaz")))

  list(
    internal_knots = internal_knots,
    boundary_knots = boundary_knots,
    coefficients = as.list(coef(fit)),
    logLik = loglik_of(fit),
    n_events = sum(dd$ev),
    log_cumhaz_baseline = pred_baseline,
    log_cumhaz_profiles = pred_profiles
  )
}

cat("Fitting cause 1...\n")
ref1 <- fit_one(1)
cat("Fitting cause 2...\n")
ref2 <- fit_one(2)

out <- list(
  time_grid = as.numeric(tgrid),
  profiles = as.list(profiles),
  cause1 = ref1,
  cause2 = ref2,
  r_version = R.version.string,
  rstpm2_version = as.character(packageVersion("rstpm2"))
)

path <- file.path(script_dir, "r_reference.json")
write_json(out, path, digits = 16, auto_unbox = TRUE, pretty = TRUE)
cat("wrote", path, "\n")
cat(sprintf("cause 1: logLik %.6f, knots [%s | %s]\n", ref1$logLik,
            paste(sprintf("%.4f", ref1$boundary_knots), collapse = ", "),
            paste(sprintf("%.4f", ref1$internal_knots), collapse = ", ")))
cat(sprintf("cause 2: logLik %.6f, knots [%s | %s]\n", ref2$logLik,
            paste(sprintf("%.4f", ref2$boundary_knots), collapse = ", "),
            paste(sprintf("%.4f", ref2$internal_knots), collapse = ", ")))
