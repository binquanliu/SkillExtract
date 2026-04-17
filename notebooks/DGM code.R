library(dplyr)
library(readr)
library(readxl)
library(stringr)
library(fuzzyjoin)  # install.packages("fuzzyjoin") if needed
library(tidyr)
library(dplyr)
library(stringr)
library(readxl)

# Set path to your folder
# folder <- "/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/rcid_quarterly_csv"
# folder <- "/Users/hanyimin/Downloads/rcid_quarterly_buckets"
folder <- "/Users/hanyimin/Downloads/rcid_monthly_categories"



# Read and combine all CSVs (exclude helper files whose names start with "_")
all_files  <- list.files(folder, pattern = "\\.csv$", full.names = TRUE)
data_files <- all_files[!str_detect(basename(all_files), "^_")]

combined <- data_files |>
  lapply(read_csv, show_col_types = FALSE) |>
  bind_rows()

# Save
write_csv(combined, file.path(folder, "combined_dataset.csv"))

cat("Combined", length(data_files), "files →", nrow(combined), "rows\n")



############ COMPARE COMPANY NAMES IN TWO FILES ############


# ── 1. Build rcid → company_name table from your CSV filenames ────────────────
# Use the same folder as the data — each per-RCID file is named "{rcid}_{name}.csv"
csv_folder <- folder

filenames <- list.files(csv_folder, pattern = "\\.csv$")
filenames <- filenames[!str_detect(filenames, "^_")]  # skip helper files like _category_columns.csv

crosswalk_panel <- tibble(filename = filenames) |>
  mutate(
    base        = str_remove(filename, "\\.csv$"),
    rcid        = as.integer(str_extract(base, "^\\d+")),
    name_panel  = str_remove(base, "^\\d+_")   # e.g. "Entergy", "American_Airlines"
  ) |>
  mutate(name_panel = str_replace_all(name_panel, "_", " ")) |>  # underscores → spaces
  select(rcid, name_panel)

# ── 2. Get unique company names from Excel ────────────────────────────────────
excel <- read_excel("/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/500_Company_AI_implementation_Raisch & Krakowski (2021) Classification.xlsx", sheet = "Sheet1")

excel_companies <- excel |>
  select(fortune_rank = Rank, name_excel = `Company name`) |>
  distinct()

# ── 3. Normalize both name columns for matching ───────────────────────────────
normalize_name <- function(x) {
  x |>
    str_to_lower() |>
    str_remove_all("\\.|,|'|'") |>
    str_replace_all("&", "and") |>
    str_remove_all("\\b(inc|corp|corporation|company|co|llc|ltd|group|holdings|
                        international|worldwide|solutions|technologies|services|
                        financial|insurance|energy|automotive|brands|industries|
                        systems|global|resorts|mutual|life|health|healthcare|
                        airline|airlines|air)\\b") |>
    str_squish()
}

crosswalk_panel  <- crosswalk_panel  |> mutate(name_norm = normalize_name(name_panel))
excel_companies  <- excel_companies  |> mutate(name_norm = normalize_name(name_excel))

# ── 4. Exact match first ──────────────────────────────────────────────────────
exact_matches <- inner_join(crosswalk_panel, excel_companies, by = "name_norm") |>
  select(rcid, name_panel, fortune_rank, name_excel)

already_matched <- c(exact_matches$rcid)

# ── 5. Fuzzy match the remainder ──────────────────────────────────────────────
panel_unmatched <- crosswalk_panel |> filter(!rcid %in% already_matched)
excel_unmatched <- excel_companies |> filter(!name_excel %in% exact_matches$name_excel)

fuzzy_matches <- stringdist_inner_join(
  panel_unmatched, excel_unmatched,
  by        = "name_norm",
  method    = "jw",
  max_dist  = 0.15,
  distance_col = "dist"
) |>
  group_by(rcid) |>
  slice_min(dist, n = 1) |>
  ungroup()

# Check what columns actually came out
print(names(fuzzy_matches))

# Then select using the actual column names
fuzzy_matches <- fuzzy_matches |>
  select(rcid, 
         name_panel  = starts_with("name_panel"),
         fortune_rank, 
         name_excel  = starts_with("name_excel"),
         dist)

# ── 6. Combine and flag for review ────────────────────────────────────────────
crosswalk_final <- bind_rows(
  exact_matches |> mutate(match_type = "exact",    dist = 0),
  fuzzy_matches |> mutate(match_type = "fuzzy")
) |>
  arrange(rcid)

# Companies in your panel with NO match in Excel
unmatched <- crosswalk_panel |>
  filter(!rcid %in% crosswalk_final$rcid) |>
  mutate(
    name_excel   = NA_character_,  # <-- typed NA
    fortune_rank = NA_integer_,    # <-- typed NA
    match_type   = "NO MATCH",
    dist         = NA_real_        # <-- typed NA
  )

crosswalk_final <- bind_rows(crosswalk_final, unmatched) |> arrange(rcid)

# ── 7. Save for review ────────────────────────────────────────────────────────
write_csv(crosswalk_final, "crosswalk_review.csv")

cat("Exact matches:  ", sum(crosswalk_final$match_type == "exact"),    "\n")
cat("Fuzzy matches:  ", sum(crosswalk_final$match_type == "fuzzy"),    "\n")
cat("No match:       ", sum(crosswalk_final$match_type == "NO MATCH"), "\n")


# Check the 4 fuzzy matches — make sure they're correct
crosswalk_final |> 
  filter(match_type == "fuzzy") |> 
  select(rcid, name_panel, name_excel, dist) |> 
  print()

# Check the 2 unmatched — see what company names they have
crosswalk_final |> 
  filter(match_type == "NO MATCH") |> 
  select(rcid, name_panel) |> 
  print()

# Also check if there are Excel companies not matched to any panel company
excel_companies |> 
  filter(!name_excel %in% crosswalk_final$name_excel) |> 
  select(fortune_rank, name_excel) |> 
  print()

# Fix the 2 no-matches directly
crosswalk_final <- crosswalk_final |>
  mutate(
    name_excel = case_when(
      rcid == 351560 ~ "Estée Lauder",   # accent in Excel name
      rcid == 941409 ~ "Core & Main",    # & got stripped
      TRUE ~ name_excel
    ),
    fortune_rank = case_when(
      rcid == 351560 ~ 279L,
      rcid == 941409 ~ 497L,
      TRUE ~ fortune_rank
    ),
    match_type = case_when(
      rcid %in% c(351560, 941409) ~ "manual",
      TRUE ~ match_type
    )
  )

# Final check
cat("Exact:  ", sum(crosswalk_final$match_type == "exact"),   "\n")
cat("Fuzzy:  ", sum(crosswalk_final$match_type == "fuzzy"),   "\n")
cat("Manual: ", sum(crosswalk_final$match_type == "manual"),  "\n")
cat("No match:", sum(crosswalk_final$match_type == "NO MATCH"), "\n")

# Save final crosswalk
write_csv(crosswalk_final, "crosswalk_final.csv")



########

# ── Helper: parse the FIRST date from a messy time string ─────────────────────
# Returns a tibble with (year, month, trans_month) using the earliest date found

parse_first_date <- function(x) {
  x <- as.character(x)

  # --- Guard against NA ------------------------------------------------------
  if (is.na(x) | x == "NA") {
    return(tibble(year = NA_integer_, month = NA_integer_, trans_month = NA_character_))
  }

  # --- Try MM/YYYY pattern first (e.g., "03/2024", "01/2023. 10/2024") --------
  mm_yyyy <- str_extract(x, "\\b(0?[1-9]|1[0-2])/(\\d{4})\\b")
  if (!is.na(mm_yyyy)) {
    parts   <- str_split(mm_yyyy, "/")[[1]]
    mo      <- as.integer(parts[1])
    year    <- as.integer(parts[2])
    return(tibble(year = year, month = mo, trans_month = sprintf("%04d-%02d", year, mo)))
  }

  # --- "Q2 2025" style → first month of the quarter --------------------------
  q_style <- str_match(x, "Q([1-4])\\s*(\\d{4})")
  if (!is.na(q_style[1])) {
    qtr  <- as.integer(q_style[2])
    year <- as.integer(q_style[3])
    mo   <- (qtr - 1L) * 3L + 1L
    return(tibble(year = year, month = mo, trans_month = sprintf("%04d-%02d", year, mo)))
  }

  # --- Season words → month --------------------------------------------------
  if (str_detect(str_to_lower(x), "summer")) {
    yr <- str_extract(x, "\\d{4}")
    if (!is.na(yr)) return(tibble(year = as.integer(yr), month = 7L,
                                  trans_month = sprintf("%04d-07", as.integer(yr))))
  }
  if (str_detect(str_to_lower(x), "early")) {
    yr <- str_extract(x, "\\d{4}")
    if (!is.na(yr)) return(tibble(year = as.integer(yr), month = 1L,
                                  trans_month = sprintf("%04d-01", as.integer(yr))))
  }

  # --- Year only (e.g., "2021", "2021-2023") → January of first year ---------
  yr <- str_extract(x, "\\d{4}")
  if (!is.na(yr)) return(tibble(year = as.integer(yr), month = 1L,
                                trans_month = sprintf("%04d-01", as.integer(yr))))

  # --- No date found ----------------------------------------------------------
  tibble(year = NA_integer_, month = NA_integer_, trans_month = NA_character_)
}


# ── Apply to Excel data ───────────────────────────────────────────────────────
excel <- read_excel("/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/500_Company_AI_implementation_Raisch & Krakowski (2021) Classification.xlsx", sheet = "Sheet1") |>
  rename(classification = `Raisch & Krakowski (2021) Classification`,
         impl_time      = `Implementation time (ideally specify to date or month)`,
         company        = `Company name`)

# Parse dates row by row
parsed_dates <- excel$impl_time |>
  lapply(parse_first_date) |>
  bind_rows()

excel <- bind_cols(excel, parsed_dates)

# ── For each company × classification, get the FIRST month ───────────────────
first_event <- excel |>
  filter(!is.na(year), classification %in% c("Automation", "Augmentation", "Both")) |>
  group_by(company, classification) |>
  slice_min(order_by = year * 100 + month, n = 1) |>   # earliest year+month
  ungroup() |>
  select(company, classification, trans_year = year, trans_month)

# Pivot wide: one row per company
first_event_wide <- first_event |>
  pivot_wider(
    names_from  = classification,
    values_from = c(trans_year, trans_month),
    names_glue  = "{.value}_{classification}"
  )
# Columns produced:
# trans_year_Automation,     trans_month_Automation     (e.g. 2023, "2023-04")
# trans_year_Augmentation,   trans_month_Augmentation
# trans_year_Both,           trans_month_Both

# ── Quick check ───────────────────────────────────────────────────────────────
glimpse(first_event_wide)
print(first_event_wide, n = 10)


# In your panel, month is already like "2021-01"
# Join crosswalk first, then:
panel <- combined

panel <- panel |>
  left_join(crosswalk_final |> select(rcid, company = name_excel), by = "rcid") |>
  left_join(first_event_wide, by = "company") |>
  group_by(rcid) |>
  mutate(TIME = row_number() - 1) |>
  ungroup()

# Helper: given a trans_month string ("2023-04"), find its TIME index
panel <- panel |>
  group_by(rcid) |>
  mutate(
    # ── Automation ──
    trans_idx_auto  = match(trans_month_Automation,  month) - 1L,
    TRANS_auto      = if_else(!is.na(trans_idx_auto) & TIME >= trans_idx_auto, 1L, 0L),
    RECOV_auto      = if_else(!is.na(trans_idx_auto) & TIME >= trans_idx_auto,
                              TIME - trans_idx_auto, 0L),
    # ── Augmentation ──
    trans_idx_aug   = match(trans_month_Augmentation, month) - 1L,
    TRANS_aug       = if_else(!is.na(trans_idx_aug)  & TIME >= trans_idx_aug,  1L, 0L),
    RECOV_aug       = if_else(!is.na(trans_idx_aug)  & TIME >= trans_idx_aug,
                              TIME - trans_idx_aug,  0L),
    # ── Both ──
    trans_idx_both  = match(trans_month_Both,        month) - 1L,
    TRANS_both      = if_else(!is.na(trans_idx_both) & TIME >= trans_idx_both, 1L, 0L),
    RECOV_both      = if_else(!is.na(trans_idx_both) & TIME >= trans_idx_both,
                              TIME - trans_idx_both, 0L)
  ) |>
  ungroup()



############## Discontinuous Growth Modeling #############


library(nlme)
library(dplyr)
library(tidyr)
library(tibble)

# ── Define outcomes and classifications ───────────────────────────────────────
# outcomes       <- c("ai_replaced_by_ai", "ai_augmented_by_ai",
#                     "ai_building_managing_ai", "ai_resistant_to_ai",
#                     "ai_transformed_by_ai")

# 6-bucket outcomes (used when folder points at rcid_quarterly_buckets):
# outcomes <- c("bucket_communication_and_creative",
#               "bucket_data_and_analytics",
#               "bucket_development_and_infrastructure",
#               "bucket_domain-specific_and_industry",
#               "bucket_enterprise_and_operations",
#               "bucket_security_and_compliance")

# 133-category outcomes: read the column-name helper written by the notebook
outcomes <- read_csv(file.path(folder, "_category_columns.csv"),
                     show_col_types = FALSE)$outcome_column

# Keep only outcomes that actually appear as columns in `panel`
outcomes <- intersect(outcomes, names(panel))

classifications <- c("aug", "auto", "both")

cat("Running DGM on", length(outcomes), "outcomes x",
    length(classifications), "classifications =",
    length(outcomes) * length(classifications),
    "outcome-class combinations\n")

# ── Storage for results ───────────────────────────────────────────────────────
results_list <- list()

# ── Main loop ─────────────────────────────────────────────────────────────────

for (outcome in outcomes) {
  for (cls in classifications) {
    
    cat("\n\n══════════════════════════════════════════════\n")
    cat("Outcome:", outcome, "| Classification:", cls, "\n")
    cat("══════════════════════════════════════════════\n")
    
    TRANS_col <- paste0("TRANS_", cls)
    RECOV_col <- paste0("RECOV_", cls)
    
    # Skip if columns don't exist
    if (!all(c(TRANS_col, RECOV_col) %in% names(panel))) {
      cat("Skipping — columns not found\n")
      next
    }
    
    dat <- panel |>
      select(rcid, month, TIME,
             TRANS = all_of(TRANS_col),
             RECOV = all_of(RECOV_col),
             DV    = all_of(outcome)) |>
      filter(!is.na(DV), !is.na(TIME))
    
    # ── STAGE 1: ICC ──────────────────────────────────────────────────────────
    icc <- NA
    tryCatch({
      s1   <- lme(DV ~ 1, random = ~ 1 | rcid, data = dat, na.action = na.omit)
      vars <- as.numeric(VarCorr(s1)[, "Variance"])
      icc  <- round(vars[1] / sum(vars), 3)
      cat("Stage 1 ICC:", icc, "\n")
    }, error = function(e) cat("Stage 1 error:", e$message, "\n"))
    
    # ── STAGE 2: Fixed effects ────────────────────────────────────────────────
    step1a <- tryCatch(
      lme(DV ~ TIME + TRANS + RECOV,
          random    = ~ 1 | rcid,
          data      = dat,
          na.action = na.omit,
          control   = lmeControl(opt = "optim")),
      error = function(e) { cat("Step1a error:", e$message, "\n"); NULL }
    )
    
    step2a <- tryCatch(
      lme(DV ~ TIME + TRANS,
          random    = ~ 1 | rcid,
          data      = dat,
          na.action = na.omit,
          control   = lmeControl(opt = "optim")),
      error = function(e) { cat("Step2a error:", e$message, "\n"); NULL }
    )
    
    step2b <- tryCatch(
      lme(DV ~ TIME + RECOV,
          random    = ~ 1 | rcid,
          data      = dat,
          na.action = na.omit,
          control   = lmeControl(opt = "optim")),
      error = function(e) { cat("Step2b error:", e$message, "\n"); NULL }
    )
    
    # ── STAGE 3: Random effects ───────────────────────────────────────────────
    stage3_1 <- tryCatch(
      lme(DV ~ TIME + TRANS + RECOV,
          random    = ~ TIME | rcid,
          data      = dat,
          na.action = na.omit,
          control   = lmeControl(opt = "optim")),
      error = function(e) { cat("Stage3_1 error:", e$message, "\n"); NULL }
    )
    
    stage3_2 <- tryCatch(
      lme(DV ~ TIME + TRANS + RECOV,
          random    = ~ TIME + TRANS | rcid,
          data      = dat,
          na.action = na.omit,
          control   = lmeControl(opt = "optim")),
      error = function(e) { cat("Stage3_2 error:", e$message, "\n"); NULL }
    )
    
    stage3_3 <- tryCatch(
      lme(DV ~ TIME + TRANS + RECOV,
          random    = ~ TIME + TRANS + RECOV | rcid,
          data      = dat,
          na.action = na.omit,
          control   = lmeControl(opt = "optim")),
      error = function(e) { cat("Stage3_3 error:", e$message, "\n"); NULL }
    )
    
    # ── Print summaries ───────────────────────────────────────────────────────
    cat("\n--- Step 1a (relative) ---\n")
    if (!is.null(step1a)) print(summary(step1a)$tTable)
    
    cat("\n--- Step 2a (no RECOV) ---\n")
    if (!is.null(step2a)) print(summary(step2a)$tTable)
    
    cat("\n--- Step 2b (no TRANS) ---\n")
    if (!is.null(step2b)) print(summary(step2b)$tTable)
    
    # Fit table
    model_list  <- list(step1a, step2a, step2b)
    model_names <- c("Step1a", "Step2a", "Step2b")
    valid       <- !sapply(model_list, is.null)
    
    fit_table <- tibble(
      model      = model_names[valid],
      neg2logLik = sapply(model_list[valid], function(m) round(-2 * logLik(m)[1], 2)),
      AIC        = sapply(model_list[valid], AIC) |> round(2),
      BIC        = sapply(model_list[valid], BIC) |> round(2)
    )
    cat("\n--- Fit indices ---\n")
    print(fit_table)
    
    # Stage 3 LRT
    cat("\n--- Stage 3 random effects LRT ---\n")
    stage3_models <- list(step1a, stage3_1, stage3_2, stage3_3)
    valid_s3      <- !sapply(stage3_models, is.null)
    if (sum(valid_s3) > 1) {
      tryCatch(
        print(do.call(anova, stage3_models[valid_s3])),
        error = function(e) cat("Stage 3 LRT error:", e$message, "\n")
      )
    }
    
    # ── Store results ─────────────────────────────────────────────────────────
    if (!is.null(step1a)) {
      results_list[[paste(outcome, cls, sep = "_")]] <- list(
        outcome        = outcome,
        classification = cls,
        icc            = icc,
        step1a         = summary(step1a)$tTable,
        fit_table      = fit_table
      )
    }
  }
}

# ── Export all Step1a coefficient tables to one CSV ──────────────────────────
coef_summary <- lapply(results_list, function(r) {
  as.data.frame(r$step1a) |>
    tibble::rownames_to_column("term") |>
    mutate(outcome = r$outcome, classification = r$classification, icc = r$icc)
}) |>
  bind_rows() |>
  select(outcome, classification, icc, term, Value, Std.Error, `t-value`, `p-value`)

write_csv(coef_summary, "/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/discontinuous_growth_results_monthly_categories.csv")
cat("\nDone! Results saved to discontinuous_growth_results_monthly_categories.csv\n")
cat("Total models:", length(results_list), "/", length(outcomes) * length(classifications), "\n")








