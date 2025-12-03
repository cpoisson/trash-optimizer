
# Goal

Create a datasets in respect to the following constraints:

Input (in ROOT_PROCESSING folder)
- Class / Category : A list of input categories
- A list of datasets already categorized
- Mapping -> Dataset categories -> target categories

Processing
Class balancing, in case of imbalanced classes (a.k.a. more images in a given folder),
use data augmentation to rebalance folders

Output: (ROOT_PROCESSED)
A balanced datasets organized as follow:
- ROOT_DIR
|_ class_0: n pictures
|_ class_1: n pictures
|_ class_2: n pictures
|_ ...



Steps:

STEP 1: Prepare ROOT_PROCESSING

Function -> download input datasets
         -> add companion file mapping existing classes to targeted one


STEP 2 : Assemble datasets
   Step 2.1 -> Parse folders
        - If mapping does not correspond raise exception (ignore class?)
        - Index paths to image in a category dic
        - Rince and repeat for all datatests in input

    Step 2.2 -> Build output dataset
        - assemble the target folder in compliance with the output specidication described above.

STEP 3 : Data Augmentation / Class balancing
