# CatSeg
Repo for model architecture code.

PFDNet.py contains the implementation for the CatSeg architecture class.

Use transformations in Transform.py to utilize the same transformations as the paper.
Use Dice_Multiclass, IoU_Multiclass and Dice_CELoss for Cataract-1k dataset. Use Dice, IoU and Dice_BCELoss for CaDIS Binary dataset. 
Cataract-1k and CaDIS (with binary annotations) can be obtained from their public dataset webpages.





-----------------------------------------------------
### Acknowledgements
The implementations in losses.py and transforms.py are adapted from publicly available repositories by Negin Ghamsarian (https://github.com/Negin-Ghamsarian). We gratefully acknowledge their contribution to the field of cataract surgical segmentation, particularly for publishing metrics and transformation pipelines that enable fair benchmarking against established datasets such as Cataract-1k and CaDIS.

