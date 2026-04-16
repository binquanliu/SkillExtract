library(dplyr)
library(tidyr)
library(readxl)
library(readr)
library(stringr)
library(purrr)
library(fuzzyjoin)
library(httr)
library(jsonlite)

# ── 1. Load all files ──────────────────────────────────────────────────────────
my_skills  <- read_csv("/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/ONET_unique_skills.csv")   # your 3003 KSAOs

skills_df     <- read_excel("/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/O*NET taxonomy/Skills.xlsx")
knowledge_df  <- read_excel("/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/O*NET taxonomy/Knowledge.xlsx")
abilities_df  <- read_excel("/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/O*NET taxonomy/Abilities.xlsx")
unspsc_df     <- read_excel("/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/O*NET taxonomy/UNSPSC Reference.xlsx")

# ── 2. Build general KSAO reference (120 elements across S/K/A) ───────────────
general_ref <- bind_rows(
  skills_df    |> distinct(`Element ID`, `Element Name`) |> mutate(domain = "Skills"),
  knowledge_df |> distinct(`Element ID`, `Element Name`) |> mutate(domain = "Knowledge"),
  abilities_df |> distinct(`Element ID`, `Element Name`) |> mutate(domain = "Abilities")
) |>
  rename(element_id = `Element ID`, element_name = `Element Name`)

cat("General KSAO elements:", nrow(general_ref), "\n")
# 35 Skills + 33 Knowledge + 52 Abilities = 120 total

# ── 3. Build UNSPSC software class reference (17 classes) ────────────────────
unspsc_ref <- unspsc_df |>
  filter(`Family Code` == 43230000) |>   # Software family only
  select(
    commodity_title = `Commodity Title`,
    class_code      = `Class Code`,
    class_title     = `Class Title`
  ) |>
  distinct()

cat("UNSPSC software commodities:", nrow(unspsc_ref), "\n")
cat("UNSPSC software classes:    ", n_distinct(unspsc_ref$class_title), "\n\n")

# ── 4. Normalize helper ────────────────────────────────────────────────────────
normalize <- function(x) {
  x |>
    str_to_lower() |>
    str_remove_all("[^a-z0-9 ]") |>
    str_replace_all("\\bsoftware\\b", "") |>
    str_replace_all("\\bsystem\\b",   "") |>
    str_replace_all("\\bplatform\\b", "") |>
    str_squish()
}

my_skills   <- my_skills   |> mutate(skill_norm = normalize(skill_name))
general_ref <- general_ref |> mutate(element_norm = normalize(element_name))
unspsc_ref  <- unspsc_ref  |> mutate(commodity_norm = normalize(commodity_title))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Match against O*NET general taxonomy (Skills, Knowledge, Abilities)
# ══════════════════════════════════════════════════════════════════════════════

# 1a. Exact match
general_exact <- my_skills |>
  inner_join(
    general_ref |> select(element_norm, element_name, element_id, domain),
    by = c("skill_norm" = "element_norm")
  ) |>
  mutate(match_type = "exact", category_source = "general_KSAO",
         category = element_name, category_detail = domain)

cat("Step 1 — Exact matches (general KSAO):", nrow(general_exact), "\n")

unmatched_1 <- my_skills |> filter(!skill_name %in% general_exact$skill_name)

# 1b. Fuzzy match against general KSAO
general_fuzzy <- stringdist_inner_join(
  unmatched_1,
  general_ref |> select(element_norm, element_name, element_id, domain),
  by           = c("skill_norm" = "element_norm"),
  method       = "jw",
  max_dist     = 0.12,   # tight threshold — general KSAOs are short phrases
  distance_col = "dist"
) |>
  group_by(skill_name) |>
  slice_min(dist, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(match_type = "fuzzy", category_source = "general_KSAO",
         category = element_name, category_detail = domain)

cat("Step 1 — Fuzzy matches (general KSAO):", nrow(general_fuzzy), "\n")

# Combine Step 1 matches
step1_matched <- bind_rows(
  general_exact |> select(skill_name, category, category_detail,
                          category_source, match_type),
  general_fuzzy |> select(skill_name, category, category_detail,
                          category_source, match_type)
)

cat("Step 1 total:", nrow(step1_matched), "→ remaining:",
    nrow(my_skills) - nrow(step1_matched), "\n\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Match remaining against UNSPSC technology classes
# ══════════════════════════════════════════════════════════════════════════════
unmatched_2 <- my_skills |> filter(!skill_name %in% step1_matched$skill_name)

# 2a. Exact match against UNSPSC commodity titles
unspsc_exact <- unmatched_2 |>
  inner_join(
    unspsc_ref |> select(commodity_norm, commodity_title, class_code, class_title),
    by = c("skill_norm" = "commodity_norm")
  ) |>
  mutate(match_type = "exact", category_source = "UNSPSC",
         category = class_title, category_detail = as.character(class_code))

cat("Step 2 — Exact matches (UNSPSC):", nrow(unspsc_exact), "\n")

unmatched_3 <- unmatched_2 |> filter(!skill_name %in% unspsc_exact$skill_name)

# 2b. Fuzzy match against UNSPSC commodity titles
unspsc_fuzzy <- stringdist_inner_join(
  unmatched_3,
  unspsc_ref |> select(commodity_norm, class_code, class_title) |> distinct(),
  by           = c("skill_norm" = "commodity_norm"),
  method       = "jw",
  max_dist     = 0.15,
  distance_col = "dist"
) |>
  group_by(skill_name) |>
  slice_min(dist, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(match_type = "fuzzy", category_source = "UNSPSC",
         category = class_title, category_detail = as.character(class_code))

cat("Step 2 — Fuzzy matches (UNSPSC):", nrow(unspsc_fuzzy), "\n")

unmatched_4 <- unmatched_3 |> filter(!skill_name %in% unspsc_fuzzy$skill_name)

# 2c. Keyword fallback for remaining software tools
keyword_rules <- tribble(
  ~pattern,                                                            ~class_title,
  "accounting|payroll|tax|billing|invoice|financial|audit",           "Finance accounting and enterprise resource planning ERP software",
  "erp|enterprise resource",                                          "Finance accounting and enterprise resource planning ERP software",
  "crm|customer relationship|sales management",                       "Data management and query software",
  "database|sql|data base|data warehouse|data mining|analytics",      "Data management and query software",
  "business intelligence|reporting|tableau|power bi",                 "Data management and query software",
  "project management|scheduling|resource planning",                  "Business function specific software",
  "human resource|hr |hris|workforce|talent|recruiting",              "Business function specific software",
  "supply chain|inventory|logistics|procurement|warehouse",           "Business function specific software",
  "helpdesk|help desk|call center|customer service|ticketing",        "Business function specific software",
  "cad|computer aided design|drafting|autocad|solidworks|revit",      "Industry specific software",
  "medical|clinical|health|ehr|emr|hospital|pharmacy",               "Industry specific software",
  "legal|compliance|regulatory|court|case management",               "Industry specific software",
  "scientific|analytical|simulation|matlab|statistical|spss|sas ",   "Industry specific software",
  "gis|geographic|mapping|spatial",                                   "Educational or reference software",
  "email|mail|outlook|collaboration|messaging|slack|teams|zoom",      "Information exchange software",
  "video conference|web conference|webinar",                          "Information exchange software",
  "security|antivirus|firewall|vpn|encryption|cyber|authentication",  "Security and protection software",
  "network|router|switch|lan|wan|wireless|cisco",                     "Networking software",
  "cloud|aws|azure|google cloud|saas|infrastructure",                 "Network management software",
  "monitor|server|active directory|itil|system management",           "System management software",
  "operating system|linux|unix|windows server",                       "Operating environment software",
  "programming|coding|python|java|javascript|github|devops",          "Development software",
  "web|html|css|php|ruby|api|agile|scrum",                           "Development software",
  "backup|storage|archival|compression|data recovery",               "Utility and device driver software",
  "document management|content management|cms|sharepoint",            "Content management software",
  "word processing|spreadsheet|presentation|office|publishing",       "Content authoring and editing software",
  "graphic|photo|image|adobe|design|illustrator|photoshop",           "Content authoring and editing software",
  "video|audio|media|streaming|editing|animation",                    "Content authoring and editing software",
  "training|learning|lms|e-learning|education",                      "Educational or reference software",
  "game|gaming|entertainment|simulation",                             "Computer game or entertainment software",
  "browser|internet|search|web application|portal",                   "Network applications software"
)

keyword_lookup <- function(skill_norm_val) {
  for (i in seq_len(nrow(keyword_rules))) {
    if (str_detect(skill_norm_val, keyword_rules$pattern[i])) {
      return(tibble(class_title = keyword_rules$class_title[i]))
    }
  }
  return(tibble(class_title = NA_character_))
}

unspsc_keyword <- unmatched_4 |>
  mutate(
    result     = map(skill_norm, keyword_lookup),
    class_title = map_chr(result, ~ .x$class_title)
  ) |>
  filter(!is.na(class_title)) |>
  select(-result) |>
  mutate(match_type = "keyword", category_source = "UNSPSC",
         category = class_title, category_detail = "keyword_assigned")

cat("Step 2 — Keyword matches (UNSPSC):", nrow(unspsc_keyword), "\n")

# Step 2 combined
step2_matched <- bind_rows(
  unspsc_exact   |> select(skill_name, category, category_detail,
                           category_source, match_type),
  unspsc_fuzzy   |> select(skill_name, category, category_detail,
                           category_source, match_type),
  unspsc_keyword |> select(skill_name, category, category_detail,
                           category_source, match_type)
)

cat("Step 2 total:", nrow(step2_matched), "\n\n")

# ── 5. Combine everything ──────────────────────────────────────────────────────
unassigned <- my_skills |>
  filter(!skill_name %in% c(step1_matched$skill_name, step2_matched$skill_name)) |>
  mutate(category = "UNASSIGNED", category_detail = NA_character_,
         category_source = "none", match_type = "none")

final <- bind_rows(step1_matched, step2_matched,
                   unassigned |> select(skill_name, category,
                                        category_detail, category_source,
                                        match_type)) |>
  arrange(category_source, category, skill_name)

# ── 6. Summary ─────────────────────────────────────────────────────────────────
cat("════════════════════════════════\n")
cat("FINAL SUMMARY\n")
cat("════════════════════════════════\n")
cat("Step 1 — General KSAO:  ", nrow(step1_matched), "\n")
cat("Step 2 — UNSPSC tech:   ", nrow(step2_matched), "\n")
cat("Unassigned:             ", nrow(unassigned),    "\n")
cat("Total:                  ", nrow(final),          "\n\n")

cat("=== Skills assigned to general KSAO categories ===\n")
step1_matched |> count(category_detail, category, sort = TRUE) |> print(n = 50)

cat("\n=== Skills assigned to UNSPSC technology classes ===\n")
step2_matched |> count(category, sort = TRUE) |> print(n = 20)

# ── 7. Save ────────────────────────────────────────────────────────────────────
write_csv(final,     "/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/O*NET taxonomy/ksao_full_mapping.csv")
write_csv(unassigned |> select(skill_name), "/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/O*NET taxonomy/ksao_unassigned_review.csv")

cat("\nSaved: ksao_full_mapping.csv\n")
cat("Saved: ksao_unassigned_review.csv\n")





############ EXAMINE UNASSIGNED KSAOs

unassigned <- read_csv("/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/O*NET taxonomy/ksao_unassigned_review.csv")

# ── PART 1: Extended keyword rules (catches ~750 of the 1815) ─────────────────
# These are more generous than the first pass

extended_keyword_rules <- tribble(
  ~pattern, ~class_title,
  
  # SAP / IBM / Oracle / major ERP suites → ERP
  "\\bsap\\b|peoplesoft|jde|baan|infor|epicor|netsuite|sage\\b|quickbooks|intuit|
   great plains|dynamics gp|dynamics nav|workday|deltek|lawson",
  "Finance accounting and enterprise resource planning ERP software",
  
  # IBM branded tools → varies by type; default to system management
  "\\bibm\\b.*tivoli|tivoli|maximo|rational|cognos|infosphere|datastage|netezza",
  "System management software",
  
  # Microsoft dev tools
  "microsoft.*visual|visual studio|visual basic|vba|vbscript|\\.net|asp\\.net|
   activex|directx|powershell|microsoft.*framework",
  "Development software",
  
  # Microsoft office/content
  "microsoft.*word|microsoft.*excel|microsoft.*powerpoint|microsoft.*outlook|
   microsoft.*onenote|microsoft.*publisher|microsoft.*visio|microsoft.*project|
   microsoft.*access|microsoft.*sharepoint|onenote|wordperfect",
  "Content authoring and editing software",
  
  # CRM / sales / marketing platforms
  "salesforce|hubspot|marketo|eloqua|pardot|act!|goldmine|siebel|sugar ?crm|
   zoho|pipedrive|constant contact|mailchimp|campaign monitor",
  "Data management and query software",
  
  # Data / BI / analytics tools
  "tableau|power bi|qlikview|qlik|microstrategy|cognos|actuate|tibco spotfire|
   looker|domo|sisense|databox|alteryx|\\bsas\\b|spss|stata|minitab|r studio|
   jupyter|anaconda|\\bknime\\b|rapidminer|weka|orange",
  "Data management and query software",
  
  # Big data / cloud data
  "hadoop|spark|kafka|hive|pig|cassandra|mongodb|redis|dynamodb|kinesis|
   redshift|snowflake|databricks|airflow|flink|impala|presto|teradata",
  "Data management and query software",
  
  # Dev frameworks / languages / tools
  "\\bjava\\b|\\bpython\\b|javascript|typescript|\\bc#\\b|\\bc\\+\\+\\b|ruby|
   php|perl|scala|swift|kotlin|golang|\\bgo\\b|rust|\\bada\\b|\\bcobol\\b|
   fortran|matlab|\\br\\b programming|jquery|react\\b|angular|vue\\.js|node\\.js|
   django|spring|flask|laravel|rails|bootstrap|graphql|ajax|ext js|ajax",
  "Development software",
  
  # DevOps / CI / containers
  "docker|kubernetes|jenkins|ansible|puppet|chef|terraform|gitlab|github|
   bitbucket|bamboo|maven|gradle|sonarqube|selenium|junit|pytest|git\\b|svn|
   subversion|concurrent versions|bugzilla|jira|confluence",
  "Development software",
  
  # Security tools
  "metasploit|nmap|wireshark|snort|nessus|burp suite|openvas|acunetix|
   tanium|logrhythm|solarwinds|nagios|zabbix|splunk|qradar|arcsight|
   mcafee|symantec|trend micro|kaspersky|truecrypt|pgp|honeypot",
  "Security and protection software",
  
  # Network management
  "solarwinds|nagios|zabbix|netcracker|infoblox|riverbed|silver peak|
   dartware|netreo|niksun|sniffer|wireshark|netscout|opennms|cacti",
  "Network management software",
  
  # CAD / engineering
  "autocad|autodesk|solidworks|catia|siemens nx|inventor|revit|solidedge|
   pro/engineer|creo|nx\\b|unigraphics|microstation|bentley|ansys|abaqus|
   nastran|hypermesh|comsol|finite element|\\bcfd\\b|cnc|mastercam|
   powermill|featurecam|geomagic|verisurf|hexagon|\\bcam\\b software",
  "Industry specific software",
  
  # Scientific / lab / research
  "labview|national instruments|labchart|adinstruments|matlab|simulink|
   mathematica|maple|\\bsas\\b|spss|\\br\\b statistical|stata|\\bsas/\\b|
   lims|laboratory information|chromatography|spectroscopy|flow cytometry|
   protein|genomic|sequence analysis|bioinformatics|dna|rna|blast\\b",
  "Industry specific software",
  
  # Medical / clinical / health
  "epic\\b|meditech|cerner|mckesson|allscripts|eclinicalworks|nextgen|
   practice fusion|drchrono|athenahealth|kareo|\\behr\\b|\\bemr\\b|
   pyxis|omnicell|medication|pharmacy|clinical|patient|\\bhis\\b|
   hospital information|radiology|dicom|pacs",
  "Industry specific software",
  
  # Legal / litigation
  "westlaw|lexisnexis|casetext|fastcase|relativity|concordance|summation|
   casemap|timematters|pclaw|prolaw|laserapp|litigat|e-discovery|ediscovery",
  "Industry specific software",
  
  # Real estate / mortgage / insurance
  "mortgage|loan origination|underwriting|fannie mae|freddie mac|
   title software|closing software|mls\\b|real estate|property management|
   yardi|appfolio|propertyware|buildium|ramquest|softpro|insurance rating|
   claims processing|policy management",
  "Industry specific software",
  
  # Construction / project estimation
  "procore|primavera|p6\\b|planswift|bluebeam|estimating|takeoff|
   bid management|construction|sage 300|viewpoint|timberline|
   on center|quick bid|cost estimat",
  "Industry specific software",
  
  # GIS / mapping / surveying
  "esri|arcgis|arcmap|arcview|arcinfo|qgis|mapinfo|erdas|envi\\b|
   lidar|surveying|gps|gnss|trimble|leica|topcon|sokkia|microsurvey",
  "Educational or reference software",
  
  # Content / media / creative
  "adobe|photoshop|illustrator|indesign|premiere|after effects|acrobat|
   lightroom|dreamweaver|flash\\b|animate|creative cloud|final cut|
   avid|pro tools|logic pro|garageband|audacity|blender|maya\\b|
   cinema 4d|houdini|nuke\\b|davinci resolve|canva",
  "Content authoring and editing software",
  
  # Collaboration / project management
  "asana|basecamp|trello|monday\\.com|wrike|smartsheet|teamwork|
   \\bjira\\b|confluence|notion\\b|airtable|clickup|workfront|
   planview|microsoft project|project management",
  "Business function specific software",
  
  # Communication / conferencing
  "zoom\\b|teams\\b|webex|gotomeeting|skype|slack\\b|google meet|
   facetime|whatsapp|telegram|signal\\b|discord|loom\\b|
   video conference|teleconference|unified communications",
  "Information exchange software",
  
  # HR / payroll / workforce
  "workday|successfactors|taleo|cornerstone|halogen|saba\\b|
   adp\\b|paychex|kronos|ultipro|bamboohr|namely|ceridian|
   \\bhris\\b|recruiting|applicant tracking|performance management",
  "Business function specific software",
  
  # Supply chain / logistics / ERP ops
  "manhattan associates|jda\\b|blue yonder|kinaxis|oracle scm|
   sap scm|sap apo|infor scm|\\b3pl\\b|dispatch|fleet management|
   route optimization|warehouse management|\\bwms\\b|\\btms\\b|
   transportation management|freight|shipping software",
  "Business function specific software",
  
  # Accounting / finance specific
  "quicken|\\bsage\\b|peachtree|blackbaud|fund accounting|
   general ledger|accounts receivable|accounts payable|
   financial reporting|budgeting software|forecasting software|
   grant management|fund management",
  "Finance accounting and enterprise resource planning ERP software",
  
  # Education / LMS
  "blackboard|canvas\\b|moodle|schoology|desire2learn|d2l\\b|
   edmodo|google classroom|instructure|brightspace|sakai|
   learning management|lms\\b|student information|sis\\b|
   powerschool|infinite campus|skyward|gradebook",
  "Educational or reference software",
  
  # Backup / storage / utility
  "backup|\\barchiv|disaster recovery|ghost\\b|acronis|veritas|
   symantec backup|commvault|veeam|zerto|barracuda|storage|
   raid\\b|san\\b|nas\\b|compression|deduplication",
  "Utility and device driver software",
  
  # Generic software type keywords (last resort)
  "scheduling software|time tracking|time attendance|timekeeping|
   workforce scheduling|staff scheduling",
  "Business function specific software",
  
  "reporting software|report generation|report writer|dashboard software",
  "Data management and query software",
  
  "document management|records management|content management|
   document imaging|document capture|ocr software",
  "Content management software",
  
  "\\bcrm\\b|customer management|contact management|sales force automation",
  "Data management and query software",
  
  "point of sale|\\bpos\\b|retail management|cash register|payment processing",
  "Business function specific software",
  
  "email software|email marketing|mail server|messaging software",
  "Information exchange software"
)

# Clean up multiline patterns
extended_keyword_rules <- extended_keyword_rules |>
  mutate(pattern = str_replace_all(pattern, "\\s+", " ") |> str_trim())

# Apply keyword matching
keyword_lookup2 <- function(skill_val) {
  s <- str_to_lower(skill_val)
  s <- str_remove_all(s, "[^a-z0-9 #.+/\\\\]")
  for (i in seq_len(nrow(extended_keyword_rules))) {
    if (str_detect(s, extended_keyword_rules$pattern[i])) {
      return(extended_keyword_rules$class_title[i])
    }
  }
  return(NA_character_)
}

unassigned <- unassigned |>
  mutate(
    class_title = map_chr(skill_name, keyword_lookup2),
    match_type  = if_else(!is.na(class_title), "keyword_extended", "none"),
    category_source = if_else(!is.na(class_title), "UNSPSC", "none"),
    category = class_title
  )

keyword_matched <- unassigned |> filter(!is.na(class_title))
still_unassigned <- unassigned |> filter(is.na(class_title))

cat("Extended keyword matches:", nrow(keyword_matched), "\n")
cat("Still unassigned:        ", nrow(still_unassigned), "\n\n")

# ── PART 2: Claude API batch classifier for remaining opaque tools ─────────────
# Define the 17 UNSPSC class labels as options
unspsc_classes <- c(
  "Business function specific software",
  "Finance accounting and enterprise resource planning ERP software",
  "Computer game or entertainment software",
  "Content authoring and editing software",
  "Content management software",
  "Data management and query software",
  "Development software",
  "Educational or reference software",
  "Industry specific software",
  "Network applications software",
  "Network management software",
  "Networking software",
  "Operating environment software",
  "Security and protection software",
  "Utility and device driver software",
  "Information exchange software",
  "System management software"
)

classify_batch <- function(skills_vec, batch_size = 50) {
  results <- character(length(skills_vec))
  batches  <- split(skills_vec, ceiling(seq_along(skills_vec) / batch_size))
  
  for (b in seq_along(batches)) {
    batch <- batches[[b]]
    cat(sprintf("Batch %d/%d (%d skills)...\n", b, length(batches), length(batch)))
    
    skill_list <- paste(seq_along(batch), batch, sep = ". ", collapse = "\n")
    classes_str <- paste(unspsc_classes, collapse = "\n")
    
    prompt <- paste0(
      "You are classifying software tools into O*NET UNSPSC technology categories.\n\n",
      "Categories (use EXACTLY these labels):\n", classes_str, "\n\n",
      "For each numbered software tool below, respond with ONLY a JSON array of objects ",
      "with keys 'n' (the number) and 'cat' (the exact category label). ",
      "No explanation, no markdown, just the JSON array.\n\n",
      "Software tools:\n", skill_list
    )
    
    resp <- POST(
      "https://api.anthropic.com/v1/messages",
      add_headers(
        "Content-Type"      = "application/json",
        "anthropic-version" = "2023-06-01"
      ),
      body = toJSON(list(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 4096,
        messages   = list(list(role = "user", content = prompt))
      ), auto_unbox = TRUE),
      encode = "json"
    )
    
    content_text <- content(resp)$content[[1]]$text
    
    tryCatch({
      parsed <- fromJSON(content_text)
      for (i in seq_len(nrow(parsed))) {
        idx <- as.integer(parsed$n[i])
        results[sum(sapply(batches[seq_len(b - 1)], length)) + idx] <- parsed$cat[i]
      }
    }, error = function(e) {
      cat("  Parse error in batch", b, ":", e$message, "\n")
    })
    
    Sys.sleep(1)  # rate limit buffer
  }
  results
}

# Run classifier on remaining unassigned
classify_batch <- function(skills_vec, batch_size = 50) {
  results <- rep(NA_character_, length(skills_vec))
  batches  <- split(skills_vec, ceiling(seq_along(skills_vec) / batch_size))
  
  for (b in seq_along(batches)) {
    batch      <- batches[[b]]
    batch_offset <- sum(sapply(batches[seq_len(b - 1)], length))
    cat(sprintf("Batch %d/%d (%d skills)...\n", b, length(batches), length(batch)))
    
    skill_list  <- paste(seq_along(batch), batch, sep = ". ", collapse = "\n")
    classes_str <- paste(unspsc_classes, collapse = "\n")
    
    prompt <- paste0(
      "Classify each numbered software tool into exactly one of these categories:\n\n",
      classes_str, "\n\n",
      "Rules:\n",
      "- Reply with ONLY a raw JSON array — no markdown, no backticks, no explanation\n",
      "- Each element: {\"n\": <number>, \"cat\": \"<exact category label>\"}\n",
      "- Use the exact category label as written above\n\n",
      "Tools to classify:\n", skill_list
    )
    
    resp <- tryCatch(
      POST(
        "https://api.anthropic.com/v1/messages",
        add_headers("Content-Type" = "application/json",
                    "anthropic-version" = "2023-06-01"),
        body = toJSON(list(
          model      = "claude-sonnet-4-20250514",
          max_tokens = 4096,
          messages   = list(list(role = "user", content = prompt))
        ), auto_unbox = TRUE),
        encode = "json"
      ),
      error = function(e) { cat("  HTTP error:", e$message, "\n"); NULL }
    )
    
    if (is.null(resp)) next
    
    # Extract raw text
    raw_text <- tryCatch(
      content(resp)$content[[1]]$text,
      error = function(e) { cat("  Could not extract text\n"); NULL }
    )
    
    if (is.null(raw_text)) next
    
    # Debug: print first 300 chars of response to see what's coming back
    cat("  Response preview:", substr(raw_text, 1, 300), "\n")
    
    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    clean_text <- raw_text |>
      str_remove("^```json\\s*") |>
      str_remove("^```\\s*")     |>
      str_remove("\\s*```$")     |>
      str_trim()
    
    # Extract just the JSON array if there's surrounding text
    json_match <- str_extract(clean_text, "\\[.*\\]")
    if (!is.na(json_match)) clean_text <- json_match
    
    tryCatch({
      parsed <- fromJSON(clean_text)
      
      # Handle both data.frame and list outputs from fromJSON
      if (is.data.frame(parsed)) {
        for (i in seq_len(nrow(parsed))) {
          idx <- as.integer(parsed$n[i])
          if (!is.na(idx) && idx >= 1 && idx <= length(batch)) {
            results[batch_offset + idx] <- parsed$cat[i]
          }
        }
      } else if (is.list(parsed)) {
        for (item in parsed) {
          idx <- as.integer(item$n)
          if (!is.na(idx) && idx >= 1 && idx <= length(batch)) {
            results[batch_offset + idx] <- item$cat
          }
        }
      }
      
      assigned <- sum(!is.na(results[batch_offset + seq_along(batch)]))
      cat(sprintf("  Assigned %d/%d\n", assigned, length(batch)))
      
    }, error = function(e) {
      cat("  Parse error:", e$message, "\n")
      cat("  Clean text was:", substr(clean_text, 1, 200), "\n")
    })
    
    Sys.sleep(1)
  }
  results
}

# ── Combine and append to main mapping ────────────────────────────────────────
newly_assigned <- bind_rows(
  keyword_matched  |> select(skill_name, category, category_source, match_type),
  still_unassigned |> select(skill_name, category, category_source, match_type)
) |> filter(!is.na(category))

# Load the original full mapping and update
full_mapping <- read_csv("/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/O*NET taxonomy/ksao_full_mapping.csv")

full_mapping_updated <- full_mapping |>
  filter(match_type != "none") |>            # keep all previously assigned
  bind_rows(newly_assigned |>                 # add newly assigned
              mutate(category_detail = "UNSPSC")) |>
  arrange(category_source, category, skill_name)

# Final summary
cat("\n═══════════════════════════════════\n")
cat("UPDATED SUMMARY\n")
cat("═══════════════════════════════════\n")
full_mapping_updated |>
  count(category_source, match_type) |>
  print()

cat("\nSkills per UNSPSC class:\n")
full_mapping_updated |>
  filter(category_source == "UNSPSC") |>
  count(category, sort = TRUE) |>
  print(n = 20)

cat("\nSkills per general KSAO domain:\n")
full_mapping_updated |>
  filter(category_source == "general_KSAO") |>
  count(category_detail, sort = TRUE) |>
  print()

# Save
write_csv(full_mapping_updated, "/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/O*NET taxonomy/ksao_full_mapping_final.csv")
cat("\nSaved: ksao_full_mapping_final.csv\n")

# Any still truly unassigned?
remaining <- full_mapping |>
  filter(match_type == "none") |>
  filter(!skill_name %in% full_mapping_updated$skill_name)
if (nrow(remaining) > 0) {
  write_csv(remaining |> select(skill_name), "/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/O*NET taxonomy/ksao_still_unassigned.csv")
  cat("Still unassigned:", nrow(remaining), "→ saved to ksao_still_unassigned.csv\n")
}





############ Matching to broader categories


# Full crosswalk: general KSAOs → broad buckets
general_ksao_to_broad <- tribble(
  ~element_name,                        ~broad_bucket,
  # Skills
  "Reading Comprehension",              "Data & analytics",
  "Active Listening",                   "Communication & creative",
  "Writing",                            "Communication & creative",
  "Speaking",                           "Communication & creative",
  "Mathematics",                        "Data & analytics",
  "Science",                            "Domain-specific & industry",
  "Critical Thinking",                  "Data & analytics",
  "Active Learning",                    "Domain-specific & industry",
  "Learning Strategies",                "Domain-specific & industry",
  "Monitoring",                         "Enterprise & operations",
  "Social Perceptiveness",              "Communication & creative",
  "Coordination",                       "Communication & creative",
  "Persuasion",                         "Communication & creative",
  "Negotiation",                        "Communication & creative",
  "Instructing",                        "Communication & creative",
  "Service Orientation",                "Communication & creative",
  "Complex Problem Solving",            "Data & analytics",
  "Operations Analysis",                "Enterprise & operations",
  "Technology Design",                  "Development & infrastructure",
  "Equipment Selection",                "Domain-specific & industry",
  "Installation",                       "Domain-specific & industry",
  "Programming",                        "Development & infrastructure",
  "Operations Monitoring",              "Enterprise & operations",
  "Operation and Control",              "Enterprise & operations",
  "Equipment Maintenance",              "Domain-specific & industry",
  "Troubleshooting",                    "Development & infrastructure",
  "Repairing",                          "Domain-specific & industry",
  "Quality Control Analysis",           "Enterprise & operations",
  "Judgment and Decision Making",       "Security & compliance",
  "Systems Analysis",                   "Development & infrastructure",
  "Systems Evaluation",                 "Development & infrastructure",
  "Time Management",                    "Enterprise & operations",
  "Management of Financial Resources",  "Enterprise & operations",
  "Management of Material Resources",   "Enterprise & operations",
  "Management of Personnel Resources",  "Communication & creative",
  # Knowledge
  "Administration and Management",      "Enterprise & operations",
  "Administrative",                     "Enterprise & operations",
  "Economics and Accounting",           "Enterprise & operations",
  "Sales and Marketing",                "Communication & creative",
  "Customer and Personal Service",      "Communication & creative",
  "Personnel and Human Resources",      "Communication & creative",
  "Production and Processing",          "Enterprise & operations",
  "Food Production",                    "Domain-specific & industry",
  "Computers and Electronics",          "Development & infrastructure",
  "Engineering and Technology",         "Development & infrastructure",
  "Design",                             "Communication & creative",
  "Building and Construction",          "Domain-specific & industry",
  "Mechanical",                         "Domain-specific & industry",
  "Mathematics",                        "Data & analytics",
  "Physics",                            "Domain-specific & industry",
  "Chemistry",                          "Domain-specific & industry",
  "Biology",                            "Domain-specific & industry",
  "Psychology",                         "Communication & creative",
  "Sociology and Anthropology",         "Communication & creative",
  "Geography",                          "Domain-specific & industry",
  "Medicine and Dentistry",             "Domain-specific & industry",
  "Therapy and Counseling",             "Communication & creative",
  "Education and Training",             "Communication & creative",
  "English Language",                   "Communication & creative",
  "Foreign Language",                   "Communication & creative",
  "Fine Arts",                          "Communication & creative",
  "History and Archeology",             "Domain-specific & industry",
  "Philosophy and Theology",            "Communication & creative",
  "Public Safety and Security",         "Security & compliance",
  "Law and Government",                 "Security & compliance",
  "Telecommunications",                 "Development & infrastructure",
  "Communications and Media",           "Communication & creative",
  "Transportation",                     "Domain-specific & industry",
  # Abilities — cognitive
  "Oral Comprehension",                 "Communication & creative",
  "Written Comprehension",              "Data & analytics",
  "Oral Expression",                    "Communication & creative",
  "Written Expression",                 "Communication & creative",
  "Fluency of Ideas",                   "Communication & creative",
  "Originality",                        "Communication & creative",
  "Problem Sensitivity",                "Data & analytics",
  "Deductive Reasoning",                "Data & analytics",
  "Inductive Reasoning",                "Data & analytics",
  "Information Ordering",               "Data & analytics",
  "Category Flexibility",               "Data & analytics",
  "Mathematical Reasoning",             "Data & analytics",
  "Number Facility",                    "Data & analytics",
  "Memorization",                       "Data & analytics",
  "Speed of Closure",                   "Data & analytics",
  "Flexibility of Closure",             "Data & analytics",
  "Perceptual Speed",                   "Data & analytics",
  "Spatial Orientation",                "Domain-specific & industry",
  "Visualization",                      "Communication & creative",
  "Selective Attention",                "Enterprise & operations",
  "Time Sharing",                       "Enterprise & operations",
  # Abilities — physical/perceptual
  "Arm-Hand Steadiness",                "Domain-specific & industry",
  "Manual Dexterity",                   "Domain-specific & industry",
  "Finger Dexterity",                   "Domain-specific & industry",
  "Control Precision",                  "Domain-specific & industry",
  "Multilimb Coordination",             "Domain-specific & industry",
  "Response Orientation",               "Domain-specific & industry",
  "Rate Control",                       "Domain-specific & industry",
  "Reaction Time",                      "Domain-specific & industry",
  "Wrist-Finger Speed",                 "Domain-specific & industry",
  "Speed of Limb Movement",             "Domain-specific & industry",
  "Static Strength",                    "Domain-specific & industry",
  "Explosive Strength",                 "Domain-specific & industry",
  "Dynamic Strength",                   "Domain-specific & industry",
  "Trunk Strength",                     "Domain-specific & industry",
  "Stamina",                            "Domain-specific & industry",
  "Extent Flexibility",                 "Domain-specific & industry",
  "Dynamic Flexibility",                "Domain-specific & industry",
  "Gross Body Coordination",            "Domain-specific & industry",
  "Gross Body Equilibrium",             "Domain-specific & industry",
  "Near Vision",                        "Domain-specific & industry",
  "Far Vision",                         "Domain-specific & industry",
  "Visual Color Discrimination",        "Domain-specific & industry",
  "Night Vision",                       "Domain-specific & industry",
  "Peripheral Vision",                  "Domain-specific & industry",
  "Depth Perception",                   "Domain-specific & industry",
  "Glare Sensitivity",                  "Domain-specific & industry",
  "Hearing Sensitivity",                "Domain-specific & industry",
  "Auditory Attention",                 "Domain-specific & industry",
  "Sound Localization",                 "Domain-specific & industry",
  "Speech Recognition",                 "Communication & creative",
  "Speech Clarity",                     "Communication & creative"
)

unspsc_to_broad <- tribble(
  ~class_title,                                                          ~broad_bucket,
  "Finance accounting and enterprise resource planning ERP software",    "Enterprise & operations",
  "Business function specific software",                                 "Enterprise & operations",
  "System management software",                                          "Enterprise & operations",
  "Data management and query software",                                  "Data & analytics",
  "Network applications software",                                       "Data & analytics",
  "Development software",                                                "Development & infrastructure",
  "Operating environment software",                                      "Development & infrastructure",
  "Networking software",                                                 "Development & infrastructure",
  "Network management software",                                         "Development & infrastructure",
  "Utility and device driver software",                                  "Development & infrastructure",
  "Security and protection software",                                    "Security & compliance",
  "Industry specific software",                                          "Domain-specific & industry",
  "Educational or reference software",                                   "Domain-specific & industry",
  "Content authoring and editing software",                              "Communication & creative",
  "Content management software",                                         "Communication & creative",
  "Information exchange software",                                       "Communication & creative",
  "Computer game or entertainment software",                             "Communication & creative"
)

# Apply to your full mapping
final_mapping <- read_csv("/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/O*NET taxonomy/ksao_full_mapping_final.csv") |>
  left_join(unspsc_to_broad,        by = c("category" = "class_title")) |>
  left_join(general_ksao_to_broad,  by = c("category" = "element_name"),
            suffix = c("", "_ksao")) |>
  mutate(
    broad_bucket = coalesce(broad_bucket, broad_bucket_ksao)
  ) |>
  select(-broad_bucket_ksao)

# Final distribution
cat("Skills per broad bucket:\n")
final_mapping |> count(broad_bucket, sort = TRUE) |> print()

write_csv(final_mapping, "/Users/hanyimin/Dropbox/Haylee/UIUC/research/AI adoption & KSAOs requirement/data/O*NET taxonomy/ksao_full_mapping_with_buckets.csv")
