#!/usr/bin/env bash

### Use above for the debug with bashdb to run properly
### Author: Han Wang
### 23-Feb-2023: Resample the image to 91 * 109 * 91 using the T1 template.




Dir_Common="/mnt/g/Backup/fMRI_dualtask/1_processed/batch_processing_spmeditor/Data/CNN/Output_Extracted"

Dirs=("$Dir_Common/dat_train" "$Dir_Common/dat_validate" "$Dir_Common/dat_test") # folders for TEST dataset

File_Ref="$Dir_Common/dat_ref/avg152T1.nii"



Dir_Train_out_resample="${Dirs[0]}/out_resample"
Dir_Validate_out_resample="${Dirs[1]}/out_resample"
Dir_Test_out_resample="${Dirs[2]}/out_resample"

mkdir $Dir_Train_out_resample
mkdir $Dir_Validate_out_resample
mkdir $Dir_Test_out_resample


for Dir in ${Dirs[@]}
do
    echo "processing folder $Dir ..."
    Ls_Dir_tmp=$(ls $Dir/*.nii.gz | xargs -n 1 basename)

    for File in $Ls_Dir_tmp
    do
        echo "resampling file ${File}..."
        3dresample -master $File_Ref -prefix $Dir/out_resample/$File -inset $Dir/$File
    done
done


    


