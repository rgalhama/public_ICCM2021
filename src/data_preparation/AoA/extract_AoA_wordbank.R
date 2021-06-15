#!/usr/bin/Rscript

# Extract Age of Acquisition data from Wordbank

# The R API:
# https://langcog.github.io/wordbankr/



##############
# First use: #
##############
#First, install these packages from command line (for devtools):
#   sudo apt-get install r-cran-rmysql
#   sudo apt-get -y install libcurl4-gnutls-dev libxml2-dev libssl-dev
#   sudo apt-get install libcurl4-openssl-dev libmysql++-dev gfortran
#   sudo apt-get install liblapack-dev -y ; sudo apt-get install liblapack3 -y ; sudo apt-get install libopenblas-base -y ; sudo apt-get install libopenblas-dev -y ;

#Then, in R:
#install.packages("magrittr") 
#install.packages("dplyr")    
#install.packages('devtools', repos='http://cran.rstudio.com/')
#install.packages("wordbankr")
#I did not use the following:
##devtools::install_github("langcog/wordbankr")
##devtools::install_github("tidyverse/glue")
###############################################################################################

#Load libraries
library(wordbankr)
library(magrittr) # need to run every time you start R and want to use %>%
library(glue)
library(dplyr)    # alternative, this also loads %>%

#source("aoa.R")

##########
# PARAMS #
##########
lang="eng"
language="English" 
proportion=0.5
measure="produces" #understands or produces
lexical_class="adjectives" # empty string if not restricted
#All available forms (WS/WG) are used
#########

##########
# PATHS  #
##########
#setwd("~/Research/WordRep/WordRep_alhama/03-raw_data/AoA/wordbank")
instr_dir=glue("/home/rgalhama/Data_Research/wordbank/instrument_data/{tolower(language)}/")
output_fname=glue("~/Research/WordRep/WordRep_alhama/03-data/AoA/wordbank/aoa_wordbank_{lang}_{measure}_prop{proportion}_{lexical_class}.csv")




#Find target words of the language 
all_items <- get_item_data()
lang_items <- all_items[grep(language, all_items$language),]
#sdf=lang_items[,1:2]
dialects=unique(lang_items$language)
forms=unique(lang_items$form)
items_ids=lang_items$item_id
rm(all_items) #save memory
#items <- get_item_data(language, form)$item_id


#Load an instrument
#  form: WS: Words and Sentences / WG: Words and Gestures
#  type: phrases, words, ...
#  lexical_category: nouns, ...

try(rm(df))
for (dialect in dialects){
    for (form in forms){
        # Remove TED forms (they are for twins, who can show different developmental trajectories)
        if (! grepl('^TEDS', form) && !grepl("WG", form)){ #
            print(glue(dialect," ",form))
            try(df<-get_instrument_data(dialect,form, administrations=TRUE, iteminfo = TRUE))
            fname=glue("instrument_data_wordbank_{dialect}_{form}.csv")
            if (exists('df')){
                try(write.table(df, fname, sep=";", quote = FALSE, col.names = TRUE, row.names = FALSE, fileEncoding = 'utf-8'))
                try(rm(df))
            }
        }
}}

#combined_data <- rbind(data,data2)


#Load instrument data from the directory and bind
filenames <- list.files(instr_dir, full.names = TRUE)
fulldata <- Reduce(rbind, lapply(filenames, read.csv, sep=";"))

#Filter by type, if specified
if (lexical_class != ""){
    data <-fulldata[grep(lexical_class, fulldata$lexical_class),]
}else{
    data <-fulldata
}
rm(fulldata)

#Estimate AoA:
#The AoA is estimated by computing the proportion of administrations for which 
#the child understands/produces (measure) each word, 
#smoothing the proportion using method, 
#and taking the age at which the smoothed value is greater than proportion.
#Params:
#measure: understands, produces
#method: glmrob, ?
#proportion: 0 <= prop <= 1
aoa=fit_aoa(data, measure = measure, method = "glmrob", proportion = proportion)

#Remove NA
aoa = aoa[!is.na(aoa$aoa),]
aoa = aoa[!is.na(aoa$uni_lemma),]

#This is not needed, since the column of interest is uni_lemma
#aoa %>%
#mutate(definition = str_replace(definition, '(.*$',""))

write.table(aoa, output_fname, sep=";", quote = FALSE, col.names = TRUE, row.names = FALSE, fileEncoding = 'utf-8') 

#See min and max AoA in this language
min(aoa$aoa)
max(aoa$aoa)



#-- old
#The administration frames have a row per kid, but no reference to the known items
#ws_adm=get_administration_data(language = "English (American)", form = "WG")


#Subselect lexical category
#wg_items_pos = wg_items[wg_items$lexical_category == pos,]
# !! some nouns are labelled as "other"













