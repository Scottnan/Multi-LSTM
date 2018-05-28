library(GCAMCTS)
library(GCAMCQT)
library(assertthat)
library(data.table)


read_tf_factor <- function(vars, from_to, freq_info = list(freq_days=1, freq_starts=1), folders = "E:/GCAMCDL_DC", .load = TRUE) {

  stopifnot(is.character(vars), !anyNA(vars), length(vars) > 0L)
  vars <- unique(vars)
  freq_days <- freq_info$freq_days
  freq_starts <- freq_info$freq_starts

  from_to <- GCAMCPUB::as_from_to(from_to)
  trade_dates <- GCAMCQT::factor_dates(from_to, freq = "daily")
  trade_dates <- trade_dates[seq(from = freq_starts, to = length(trade_dates), by = freq_days)]

  GCAMCPUB::log_debug("read defined/factor with date_from = ", from_to$from, ", date_to = ", from_to$to, "...")
  factor <- list()
  dat <-
    GCAMCQT:::run_loop(
      vars = vars,
      fun = function(var, vars) {
        withCallingHandlers({
          dat <- read_tf_factor_impl(var, trade_dates, folders)
          if (isTRUE(.load)) factor[[var]] <- dat
          dat
        }, error = function(e) {
          stop("fail to read factor ", var ,":\n", e$message, call. = FALSE)
        })
      },
      pb_msg_prefix = "reading",
      info_msg = "read defined factor [{{progress}}] {{var}}...",
      pb_msg_level = "read_hfd_factor"
    )
  invisible(as.environment(dat))
}


read_tf_factor_impl <- function(var, trade_dates, folders) {

  factor_folder <- choose_folder(var, folders)
  storage_info <- factor_storage(var, trade_dates, factor_folder)
  dt <-
    purrr::map(
      storage_info$FILE_PATH,
      function(x) {
        if (file.exists(x)) {
          dt <- fst::read_fst(x, as.data.table = TRUE)
        } else {
          dt <- data.table()
        }
      }
    ) %>%
    data.table::rbindlist(.) %>%
    data.table::setkeyv(., c("INNER_CODE", "DATE"))
  stopifnot(GCAMCPUB::is_pk_dt(dt))
  invisible(dt)
}


factor_storage <- function(var, trade_dates, factor_folder) {

  date_seq <- trade_dates
  res <- list()

  for (i in 1L:(length(date_seq))) {
    from <- date_seq[i]
    to <- date_seq[i]
    file_name <- fst_file_name(var, from, to)
    path <- file.path(factor_folder, var, file_name)
    res[[i]] <-
      data.table(
        FILE_PATH = path,
        DATE_FROM = from,
        DATE_TO = to
      )
  }
  res <- data.table::rbindlist(res)
}


fst_file_name <- function(var, from, to) {

  stopifnot(is.string(var), is.date(from), is.date(to))
  paste0(paste(var, format(from, "%Y%m%d"), format(to, "%Y%m%d"), sep = "_"), ".fst")
}


choose_folder <- function(var, folders) {

  res <- purrr::map(setNames(folders, folders), function(x) {
    file_names <- dir(x)
    res <- any(var %in% file_names)
  }) %>% unlist(.)
  choosen_folder <- folders[res]

  stopifnot(length(choosen_folder) == 1)
  choosen_folder
}


tf_factor_tbl <- function(vars, from_to, value_col, freq_info = list(freq_days=1, freq_starts=1), need_rtn=TRUE, folders = "E:/GCAMCDL_DC") {

  from_to <- GCAMCPUB::as_from_to(from_to)
  universe <- GCAMCQT::factor_universe(from_to, freq = "daily")
  data.table::alloc.col(universe, n = 1000)

  values <- read_tf_factor(vars, from_to, freq_info, folders, .load = FALSE)
  universe <- universe[DATE %in% values[[vars[1]]]$DATE]
  for (var in vars) {
    value <- values[[var]][universe, c(value_col), roll = FALSE, with = FALSE]
    set(universe, j = var, value = value)
  }

  if(need_rtn) universe[, "fwd_rtn" := ashare_fwd_rtns(INNER_CODE, DATE)[,.(RTN)]]
  universe[]
}
