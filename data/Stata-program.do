***********************************************************************************************************
****************************** STATA program for the regressions ******************************************
******************************* (Stata Version 11.2 for Windows)    ***************************************
***********************************************************************************************************



**************************************************************
********* Variables for baseline regressions *****************
**************************************************************
gen s=score-75
gen s2=s^2
gen s3=s^3

gen streat=s*treat
gen streat2=s2*treat
gen streat3=s3*treat

gen notreat=1-treat
gen snotreat=s*notreat
gen snotreat2=s2*notreat
gen snotreat3=s3*notreat




********************************************************************
******* 1.  Baseline  (Table 3-4) **********************************
********************************************************************

***full sample pol=0 (polynomial of zero degree)
reg  INVSALES  treat					, cluster(score)
est store A
est stats
***pol=1 (polynomial of degree one)
reg  INVSALES treat snotreat streat               , cluster(score)
est store A
est stats
***pol=2 (polynomial of degree two)
reg  INVSALES treat snotreat snotreat2 streat streat2   , cluster(score)
est store A
est stats
***pol=3 (polynomial of degree three)
reg  INVSALES treat snotreat snotreat2 snotreat3 streat streat2 streat3 , cluster(score)
est store A
est stats

*local estimates 50%
reg  INVSALES treat  			if score>51 & score<81, cluster(score)
est store A
est stats
reg  INVSALES treat snotreat streat          if score>51 & score<81, cluster(score)
est store A
est stats
reg  INVSALES treat snotreat snotreat2 streat streat2 if score>51 & score<81, cluster(score)
est store A
est stats

*local estimates 35%
reg  INVSALES treat  			if  score>65 & score<79, cluster(score)
est store A
est stats
reg  INVSALES treat snotreat streat          if score>65 & score<79, cluster(score)
est store A
est stats
reg  INVSALES treat snotreat snotreat2 streat streat2 if score>65 & score<79, cluster(score)
est store A
est stats



*******************************************************************************
********* 2. Small-large firms  (Table 5) *************************************
*******************************************************************************


** variables for the regressions
gen ssmall=s*smallm
gen ssmall2=s2*smallm
gen ssmall3=s3*smallm
gen slarge=s*largem
gen slarge2=s2*largem
gen slarge3=s3*largem

gen treatsmall=treat*smallm
gen streatsmall=s*treat*smallm
gen streatsmall2=s2*treat*smallm
gen streatsmall3=s3*treat*smallm
gen treatlarge=treat*largem
gen streatlarge=s*treat*largem
gen streatlarge2=s2*treat*largem
gen streatlarge3=s3*treat*largem


******* regressions ******************
******* full sample
reg INVSALES  largem treatsmall treatlarge			, cluster(score)
est store A
est stats
reg INVSALES largem treatsmall treatlarge ssmall slarge streatsmall streatlarge   , cluster(score)
est store A
est stats
reg INVSALES largem treatsmall treatlarge ssmall slarge ssmall2 slarge2 streatsmall streatlarge streatsmall2 streatlarge2  , cluster(score)
est store A
est stats
reg INVSALES largem treatsmall treatlarge ssmall slarge ssmall2 slarge2 ssmall3 slarge3 streatsmall streatlarge streatsmall2 streatlarge2 streatsmall3 streatlarge3  , cluster(score)
est store A
est stats

*local estimates 50%
reg INVSALES  largem treatsmall treatlarge	if score>51 & score<81, cluster(score)
est store A
est stats
reg INVSALES largem treatsmall treatlarge ssmall slarge streatsmall streatlarge   if score>51 & score<81, cluster(score)
est store A
est stats
reg INVSALES largem treatsmall treatlarge ssmall slarge ssmall2 slarge2 streatsmall streatlarge streatsmall2 streatlarge2  if score>51 & score<81, cluster(score)
est store A
est stats




*local estimates 35%
reg INVSALES  largem treatsmall treatlarge	if score>65 & score<79, cluster(score)
est store A
est stats
reg INVSALES largem treatsmall treatlarge ssmall slarge streatsmall streatlarge   if score>65 & score<79, cluster(score)
est store A
est stats
reg INVSALES largem treatsmall treatlarge ssmall slarge ssmall2 slarge2 streatsmall streatlarge streatsmall2 streatlarge2  if score>65 & score<79, cluster(score)
est store A
est stats


*******************************************************************************
********* 3. Coverage ratio (Table 6) *****************************************
*******************************************************************************
** high and low  coverage ratio
gen     high=1     if CR>.4043 & treat==1
replace high=0     if CR<=.4043 | treat==0
gen     low=1      if CR<=.4043 & treat==1
replace low=0      if high==1 | treat==0

** variables for regressions
gen treath=treat*high
gen streath=s*treath
gen streath2=s2*treath
gen streath3=s3*treath
gen treatl=treat*low
gen streatl=s*treatl
gen streatl2=s2*treatl
gen streatl3=s3*treatl


***************** regressions **************************
******* full sample
reg INVSALES treatl treath 				, cluster(score)
est store A
est stats
reg INVSALES treatl treath snotreat streatl streath           , cluster(score)
est store A
est stats
reg INVSALES treatl treath snotreat snotreat2 streatl streatl2 streath streath2, cluster(score)
est store A
est stats
reg INVSALES treatl treath snotreat snotreat2 snotreat3 streatl streatl2 streatl3 streath streath2 streath3    , cluster(score)
est store A
est stats

*local estimates 50%
reg INVSALES treatl treath 				if score>51 & score<81, cluster(score)
est store A
est stats
reg INVSALES treatl treath snotreat streatl streath  if score>51 & score<81, cluster(score)
est store A
est stats
reg INVSALES treatl treath snotreat snotreat2 streatl streatl2 streath streath2      if score>51 & score<81, cluster(score)
est store A
est stats

*local estimates 35%
reg INVSALES treatl treath 		if score>65 & score<79, cluster(score)
est store A
est stats
reg INVSALES treatl treath snotreat streatl streath  if score>65 & score<79, cluster(score)
est store A
est stats
reg INVSALES treatl treath snotreat snotreat2 streatl streatl2 streath streath2      if score>65 & score<79, cluster(score)
est store A
est stats


*******************************************************************************
**** 4. Age: young firms (=fchighm) and old firms (=fclowm) *****  (Table 6) **
*******************************************************************************

** young and old firms
gen     fclowm=1    if AGE<1987.081      & AGE!=.
replace fclowm=0    if AGE>=1987.081     & AGE!=.
gen     fchighm=1   if AGE>=1987.081     & AGE!=.
replace fchighm=0   if AGE<1987.081      & AGE!=.

** variables for regressions
gen sfclow=s*fclowm
gen sfclow2=s2*fclowm
gen sfclow3=s3*fclowm
gen sfchigh=s*fchighm
gen sfchigh2=s2*fchighm
gen sfchigh3=s3*fchighm

gen treatfclow=treat*fclowm
gen streatfclow=s*treat*fclowm
gen streatfclow2=s2*treat*fclowm
gen streatfclow3=s3*treat*fclowm
gen treatfchigh=treat*fchighm
gen streatfchigh=s*treat*fchighm
gen streatfchigh2=s2*treat*fchighm
gen streatfchigh3=s3*treat*fchighm

******* regressions ******************
******* full sample
reg INVSALES  fclowm treatfchigh treatfclow 			, cluster(score)
est store A
est stats
reg INVSALES fclowm treatfchigh treatfclow sfchigh sfclow streatfchigh streatfclow   , cluster(score)
est store A
est stats
reg INVSALES fclowm treatfchigh treatfclow sfchigh sfclow sfchigh2 sfclow2 streatfchigh streatfclow streatfchigh2 streatfclow2  , cluster(score)
est store A
est stats
reg INVSALES fclowm treatfchigh treatfclow sfchigh sfclow sfchigh2 sfclow2 sfchigh3 sfclow3 streatfchigh streatfclow streatfchigh2 streatfclow2 streatfchigh3 streatfclow3  , cluster(score)
est store A
est stats

*local estimates 50%
reg INVSALES  fclowm treatfchigh treatfclow	if score>51 & score<81, cluster(score)
est store A
est stats
reg INVSALES fclowm treatfchigh treatfclow sfchigh sfclow streatfchigh streatfclow   if score>51 & score<81, cluster(score)
est store A
est stats
reg INVSALES fclowm treatfchigh treatfclow sfchigh sfclow sfchigh2 sfclow2 streatfchigh streatfclow streatfchigh2 streatfclow2  if score>51 & score<81, cluster(score)
est store A
est stats

*local estimates 35%
reg INVSALES  fclowm treatfchigh treatfclow	if score>65 & score<79, cluster(score)
est store A
est stats
reg INVSALES fclowm treatfchigh treatfclow sfchigh sfclow streatfchigh streatfclow   if score>65 & score<79, cluster(score)
est store A
est stats
reg INVSALES fclowm treatfchigh treatfclow sfchigh sfclow sfchigh2 sfclow2 streatfchigh streatfclow streatfchigh2 streatfclow2  if score>65 & score<79, cluster(score)
est store A
est stats
