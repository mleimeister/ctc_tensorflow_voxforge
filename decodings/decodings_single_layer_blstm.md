## Example decodings

To monitor the evolution of the network during training, the first batch was decoded every 10 epochs and printed to be compared with the target transcription.

The original target transcription of the first batch is:

**you have all the advantage and each year something happened and i did not go the dirk mentioned by wolf larsen rested in its sheath on my hip an altruistic act is an act performed for the welfare of others**

After 10 epochs, with a label error rate (LER) of 0.779, the decoded output contains a lot of vocals and much fewer labels than the target, meaning a lot of 'blanks' are predicted. These are favoured by the model first due to the setup of the CTC error, which puts a 'blank' between every character. Predicting a lot of 'blanks' therefore decreases the error quickly in the beginning:

**e a a a a a a a a i i i atei onine e ei a e eteao itit o i a e eto e e e etfo o t** 

After 30 epochs, LER = 0.585, the basic structure for some words is visible, e.g. 'others', 'act', 'performed':

**o o v t t a et e e an id d ot othei henoiond e wowlrerestean a rirtic cis nact rerfrmad o he elfa othrs**


40 epochs, LER = 0.511:

**a a aea a a e a a  ea e e e a a a athe dir eid b wol lasn restean astristic act i an act performed fo the welfafofo othrs**


50 epochs, LER = 0.449, the last sentence ('an altruistic act...') only misses some single letters:

**afae e asedainarandn diono othe dirk mentioned by wolf laren restean alruistic act is an act performed fo the welfare of others**


60 epochs, LER = 0.420:

**tav a ea e a ad i did not gothe dirk mentioned by wolf larsen reste an altruistic act is an act performed for the welfare of others**


90 epochs, LER = 0.323:

**u have a annd each eae eld and i did not gothe dirk mentioned by wolf larsen reste an altruistic act is an act performed for the welfare of others**


120 epochs, LER = 0.252:

**ou have al adwantaend each ea eheh h hp ened abda and ididnot gothe dirk mentioned by wolf larsen reste an altruistic act is an act performed for the welfare of others**


Epoch 150, LER = 0.180:

**you have al advantaeand eachyea so atn pened and i did not gothe dirk mentioned by wolf larsen reste an altruistic act is an act performed for the welfare of others**


Epoch 170, LER = 0.156, some spaces are missing ('theadvantageand...')

**you have al theadvantageand each year sotinp hapened and i did not gothe dirk mentioned by wolf larsen reste an altruistic act is an act performed for the welfare of others**

190 epochs, LER = 0.117, most of the predicted words are correct, the main part of the error seems to come from missing words, where there are still blanks predicted instead:

**you have al the advantageand each year soething hapened and i did not gothe dirk mentioned by wolf larsen reste an altruistic act is an act performed for the welfare of others**



 
 
 
 
 
 
 
