10/17/22
okay, so SCI just doesn't seem to work! I've messed with both PCA to certain dimensions, and with instead using all 32 channesl, and with a couple different alignment techniques and all doesn't seem to generalize well to the entire test set. Also, when zooming in on certain sections, even the good gaits it doesn't seem to work.

However, I think part of this is likely just poor DLC stuff. We also need to find a way to annotate when the rat is running well. and only use those to train. I think this might help a lot? but i'm not sure. 

Regarding future steps for SCI, we shouuld try on a few more datasets, but I'm optimistic about either uising a really good injureud rat as a preloaded decoder, or using a healing rat that is slowly regaining funciton of its legs to help. vs pre-injury Rat. 

also need to try alignment on forelimb on two pre-sci datasets from Grant

I also did some double checking of my previous results. All seems good.

tomorrow hopefully make some nice figures.

10/11/22
added plot per percent for Regression Decoder Evaluation.ipynb

need to understand why VAF score for stitched SCI plot seems high, when actual plots don't seem reasonable.

and plotting actual live video frames fo angle peaks  

10/6/22

i think r_decoder works at getting well with less training data (although the practical implications of the higher score are a little tricky for me to tell). I think more importantly would be to start figuring out how well it works pre vs post sci

Regression Decoder Evaluation.ipynb in scripts is the most final version of it so far i think

