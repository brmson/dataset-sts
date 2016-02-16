## Use this R script as follows:

## R --no-save --slave --vanilla --args your-scores gold < sick_evaluation.R 

## where your-scores is the file with your system output and gold is
## the gold standard file (for example, the SICK_trial.txt and
## SICK_train.txt files we already released).

## Your file must contain the following 3 tab-delimited columns:

## -pair_ID (the ids, that should match those in the relevant the
## -trial, train or test data),

## - entailment_judgment (predictions of your system for the
## entailment sub-task; possible values: ENTAILMENT, CONTRADICTION,
## NEUTRAL) and

## - relatedness_score (numerical predictions of your system for the
## sentence relatedness sub-task).

## Note that the first line of the file must be a "header" naming the
## 3 columns exactly with the 3 strings above (pair_ID,
## entailment_judgment and relatedness_score).

## The order of the columns and rows does not matter: of course, the
## ids must match those in the relevant data sets.

## If you do not participate in the entailment or relatedness task,
## please provide a column of NA (by that, we mean that you should
## literally enter the string NA in each row).
## Note that, for either subtask you want to evaluate, you must 
## provide a value for each test pair: if your scores contain 1) NAs 
## or 2) missing values, the script will either ignore the subtask 
## or return an error. 

## The script returns system (percentage) accuracy for the entailment
## task and Pearson and Spearman correlation scores and mean squared
## error for the relatedness task.


ifile = commandArgs()[6];
gold = commandArgs()[7];

read.delim(ifile, sep="\t", header=T, stringsAsFactors=F) -> score;
read.delim(gold, sep="\t", header=T) -> gold;

score <- score[order(score$pair_ID), ];
gold <- gold[order(gold$pair_ID), ];



if (FALSE %in% (score$pair_ID==gold$pair_ID)){
	print("ERROR: pair IDs in score set and gold set do not correspond")
	quit()
	}


print(paste("Processing ", ifile, sep=""));

if (TRUE %in% is.na(score$entailment_judgment)){
	print("No data for the entailment task: evaluation on relatedness only")
	}else if (FALSE %in% (unique(score$entailment_judgment) %in% unique(gold$entailment_judgment))){
	print("ERROR: wrong level(s) in entailment judgments")
	}else{
	accuracy <- sum(score$entailment_judgment == gold$entailment_judgment) / length(score$entailment_judgment)*100
	print(paste(paste("Entailment: accuracy ", accuracy, sep=""),"%",sep=""))
	}
	
if (TRUE %in% is.na(score$relatedness_score)){
	print("No data for the relatedness task: evaluation on entailment only ")
	}else if (is.numeric(score$relatedness_score)==FALSE){
	print("ERROR: wrong format for relatedness scores")
	}else{
	pearson <- cor(score$relatedness_score, gold$relatedness_score)
	print(paste("Relatedness: Pearson correlation ", pearson, sep=""))
	spearman <- cor(score$relatedness_score, gold$relatedness_score, method = "spearman")
	print(paste("Relatedness: Spearman correlation ", spearman, sep=""))
	MSE <- sum((score$relatedness_score - gold$relatedness_score)^2) / length(score$relatedness_score)
	print(paste("Relatedness: MSE ", MSE, sep=""))
	}

quit()	
