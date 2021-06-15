# Extracts child-directed speech from CHILDES
# @author: __rgalhama__

# Uses the R API in the childes-db project:
# http://childes-db.stanford.edu/api.html


##############
# First use: #
##############
#First, install these packages from command line:
#   sudo apt-get install r-cran-rmysql
#   sudo apt-get -y install libcurl4-gnutls-dev libxml2-dev libssl-dev

#Then, in R:
install.packages('devtools', repos='http://cran.rstudio.com/')
devtools::install_github("langcog/childesr")
###############################################################################################


##########
# PARAMS #
##########
lang="eng"
age_range=c(0,60)
#ages in database: 0 to 228 (19 years!!!)
#########


#Database version: 2018.1 (doublecheck if it's this one or an earlier one)
#Load database
library(childesr)
library(glue)     #for string formatting

# Select child directed speech in the given language and age range
utterances = get_utterances(language=lang, age=age_range, role_exclude = "Target_Child")

#Export child-directed speech utterances (gloss)
gloss=utterances$gloss
fname=glue("{lang}_{age_range[1]}_{age_range[2]}_cds_utterances.txt")
write.table(gloss, fname, quote = FALSE, col.names = FALSE, row.names = FALSE, fileEncoding = 'utf-8') 

#Export child-directed speech stemmed sentences
stem=utterances$stem
fname=glue("{lang}_{age_range[1]}_{age_range[2]}_cds_stem.txt")
write.table(stem, fname, quote = FALSE, col.names = FALSE, row.names = FALSE, fileEncoding = 'utf-8') 

#Export part-of-speech on different file
pos=utterances$part_of_speech
fname=glue("{lang}_{age_range[1]}_{age_range[2]}_cds_pos.txt")
write.table(pos, fname, quote = FALSE, col.names = FALSE, row.names = FALSE, fileEncoding = 'utf-8') 


