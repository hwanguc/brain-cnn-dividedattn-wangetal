#!/usr/bin/env bash

### Use above for the debug with bashdb to run properly
### Adapted from the script writted by Xiaoxiao Wang (Wang et al., 2019)
### Author: Han Wang
### 24-Feb-2023:
### 23-Feb-2023: The script no longer selects 27 random images for the training set, this will be performed during training.
### 23-Feb-2023: Added resampling of the images to 91 x 109 x 91 and striping of the skull.
### 06-Feb-2023: The script now outputs the training (18 subj), validation (2 subj), and testing datasets (5 subj).
### 30-Jan-2023: Changing the script to extract images per task per block and output all files to one folder.

### basic setups

#### setup freesurfer
cd ~/freesurfer
pwd
export FREESURFER_HOME=$HOME/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh


#### folders for all participants' preprocessed data of the dual-task.
Dir_Common="/mnt/g/Backup/fMRI_dualtask/1_processed/batch_processing_spmeditor/Data"

Dir_TMP="$Dir_Common/CNN/Output_Extracted/tmp" # a temporary folder
Dir_Output="$Dir_Common/CNN/Output_Extracted" # folder for output (31 volumes maximum, NOT yet extracted for training, etc.)
File_Ref="$Dir_Output/dat_ref/avg152T1.nii"




Dat_type_txt=$Dir_TMP/train_va_test.txt
tr=`sed -n "2,1p" $Dat_type_txt|awk '{print $2}'`
va=`sed -n "7,1p" $Dat_type_txt|awk '{print $2}'`
te=`sed -n "3,1p" $Dat_type_txt|awk '{print $2}'`


Dir_All="$Dir_Common/CNN/Output_Extracted/dat_all" # folder for all dataset output (first 27 volumnes)ech
Dir_Train="$Dir_Common/CNN/Output_Extracted/dat_train" # folder for training dataset output (random 27 volumes)
Dir_Validate="$Dir_Common/CNN/Output_Extracted/dat_validate" # folder for validation dataset output (first 27 volumes)
Dir_Test="$Dir_Common/CNN/Output_Extracted/dat_test" # folder for testing dataset output (first 27 volumnes)ech


Subj_Seq=(`seq -f '%02g' 1 25`) #the number sequence of 25 participants, stored as an array
Subj_ID=(${Subj_Seq[@]/#/sub-}) #add a prefix to each of the element
Num_Subj=${#Subj_ID[@]}
Num_Run=0 # this will be updated in the loop as runs across subj are uneven.
Num_Cond=4
Num_RepCond=2

Dig_TR_Sec=1.3
Dig_DurPostTask_Sec=9.1
Dig_DurPostTask_Sec_St=3.9


#### Loop through the subject folder to output the extracted files to the output folder

### CHANGE the iSubj= to the ID you want to start with (need to put ID-1)

for (( iSubj=0; iSubj<$Num_Subj; iSubj++ ))
do
    Subj_tmp=${Subj_ID[$iSubj]}
    mkdir $Dir_TMP/$Subj_tmp
    Dir_tmp_subj="$Dir_TMP/$Subj_tmp"
    echo "Processing $Subj_tmp..."


    echo "Copying the unsmoothed func images to tmp folder..."
    cp $Dir_Common/$Subj_tmp/func/wrsub* $Dir_tmp_subj # copy unsmoothed image files to the temp subj folder
    echo "Copying the timing files to tmp folder..."
    cp $Dir_Common/$Subj_tmp/func/timing/* $Dir_tmp_subj # copy all timing files to the temp subj folder

    # sub-06 and sub-12 only have 5 runs
    if [ $iSubj -eq 5 ] || [ $iSubj -eq 11 ]
    then
        Num_Run=5
    else
        Num_Run=6
    fi

    # looping through runs and conds and condreps
    for (( iRun=1; iRun<=$Num_Run; iRun++ ))
    do

        File_InputNii=$Dir_tmp_subj/wr$(echo $Subj_tmp | tr -d -)-run${iRun}_bold.nii

        # Get the index of the last volume
        Ind_Last=`3dinfo -n4 $File_InputNii`
        Ind_Last=`echo $Ind_Last | awk '{print $NF}'`
        Ind_Last=$(echo "$Ind_Last-1" |bc)

        # convert the current run nii to 3d nii AND perform skull stripping

        RunSplitDir=$Dir_tmp_subj/Run${iRun}Split
        RunStripDir=$Dir_tmp_subj/Run${iRun}Strip
        RunStripMergeDir=$Dir_tmp_subj/Run${iRun}StripMerge
        RunStripedMergeResampledDir=$Dir_tmp_subj/Run${iRun}StripMergeResample
        mkdir $RunSplitDir
        mkdir $RunStripDir
        mkdir $RunStripMergeDir
        mkdir $RunStripedMergeResampledDir
        

        echo "Started pre-processing Run ${iRun}..."

        for ((iSptIdx=0; iSptIdx<=$Ind_Last; iSptIdx++ ))
        do
            echo "Converting 4d nifti to 3d nifti..."
            echo "Generating idx ${iSptIdx} out of ${Ind_Last} indices..."
            #File_InputNiiCurrent=${File_InputNii[$iSptIdx]}

            #echo ${File_InputNii[$iSptIdx]}

            3dTcat -prefix $RunSplitDir/wr$(echo $Subj_tmp | tr -d -)-run${iRun}_idx$iSptIdx.nii.gz ${File_InputNii}[$iSptIdx]

            echo "Stripping off the skull for idx ${iSptIdx} out of ${Ind_Last} indices..."
            mri_synthstrip -i $RunSplitDir/wr$(echo $Subj_tmp | tr -d -)-run${iRun}_idx$iSptIdx.nii.gz -o $RunStripDir/wr$(echo $Subj_tmp | tr -d -)-run${iRun}_stripped_idx$iSptIdx.nii.gz

        done

        # Concatenate all 3d niis back to a 4d one

        echo "Merging 3d nifti to a 4d one..."
        3dTcat -prefix $RunStripMergeDir/wr$(echo $Subj_tmp | tr -d -)-run${iRun}_stripmerged.nii.gz $RunStripDir/wr$(echo $Subj_tmp | tr -d -)-run${iRun}_stripped_idx* -tr $Dig_TR_Sec

        # Resample all images to standard 91 x 109 x 91 MNI space

        echo "Resample the 4d nifti into a 91x109x91 space..."
        3dresample -master $File_Ref -prefix $RunStripedMergeResampledDir/wr$(echo $Subj_tmp | tr -d -)-run${iRun}_stripmergedresampled.nii.gz -inset $RunStripMergeDir/wr$(echo $Subj_tmp | tr -d -)-run${iRun}_stripmerged.nii.gz

        File_InputNii_Processed=$RunStripedMergeResampledDir/wr$(echo $Subj_tmp | tr -d -)-run${iRun}_stripmergedresampled.nii.gz
        

        for ((iCond=1; iCond<=$Num_Cond; iCond++ ))
        do
            for ((iRepCond=1; iRepCond<=$Num_RepCond; iRepCond++ ))
            do
                echo "Started separating files into conditions..."
                echo "Reading RepCond${iRepCond} in run${iRun}_cond${iCond}.txt"

                # Get the begining time and duration of the task, as well as the corresponding num of volumns
                Dig_TaskBegin_Sec=`sed -n "$iRepCond,1p" $Dir_tmp_subj/run${iRun}_cond${iCond}.txt|awk '{print $1}'`

                ## The FUCKING bash doesn't suport floating-point arithmetic!!! So we need to round the beginsec to the nearest integer for the if condition below
                #Dig_TaskBegin_Sec_Int=$(echo "($Dig_TaskBegin_Sec+0.5)/1" |bc)

                Dig_DurTask_Sec=`sed -n "$iRepCond,1p" $Dir_tmp_subj/run${iRun}_cond${iCond}.txt|awk '{print $2}'`

                Num_DurTask_Ind=$(echo "((($Dig_DurTask_Sec+$Dig_DurPostTask_Sec)/$Dig_TR_Sec)+0.5)/1" |bc)

                #if [[ $Dig_TaskBegin_Sec_Int -gt 288 ]]
                #then
                #    Num_DurTask_Ind=$(echo "((($Dig_DurTask_Sec+$Dig_DurPostTask_Sec_St)/$Dig_TR_Sec)+0.5)/1" |bc)
                #else
                #    Num_DurTask_Ind=$(echo "((($Dig_DurTask_Sec+$Dig_DurPostTask_Sec)/$Dig_TR_Sec)+0.5)/1" |bc)
                #fi


                
                # Get the volumn indices
                Ind_TaskBegin=$(echo "scale=0;$Dig_TaskBegin_Sec/$Dig_TR_Sec" |bc)
                Ind_TaskEnd=$(echo "$Ind_TaskBegin+$Num_DurTask_Ind-1" |bc)

                if [[ $Ind_TaskEnd -gt $Ind_Last ]]
                then
                    Ind_TaskEnd=$Ind_Last
                    Num_DurTask_Ind=$(echo "$Ind_TaskEnd-$Ind_TaskBegin+1" |bc)
                fi

                # Extract the volumes
                echo "Extracting volumes for $Subj_tmp Run ${iRun} Cond ${iCond} RepCond ${iRepCond}..."

                File_OutputMax31=$Dir_All/$Subj_tmp'_Run-'$(echo "$iRun" |bc)'_Cond-'$(echo "$iCond" |bc)'_RepCond-'$(echo "$iRepCond" |bc)'_Vol-'$(echo "$Num_DurTask_Ind" |bc).nii.gz
                3dTcat -prefix $File_OutputMax31 \
                        $File_InputNii_Processed'['$Ind_TaskBegin'..'$Ind_TaskEnd']'
                echo "Done Segmentation!"

                # Copy files to training, validation, and testing folders

                Dat_type=`sed -n "$(echo "$iSubj+1" |bc),1p" $Dat_type_txt|awk '{print $2}'`
                echo "Dat_type is $Dat_type"

                if [[ $Num_DurTask_Ind -gt 26 ]]
                then
                    
                    echo "Copying file to the $Dat_type folder..."

                    if [[ "$Dat_type" = "$te" ]]
                    then 
                        Dir_Output_27=$Dir_Test
                    elif [[ "$Dat_type" = "$va" ]]
                    then
                        Dir_Output_27=$Dir_Validate
                    else
                        Dir_Output_27=$Dir_Train
                    fi

                    cp $File_OutputMax31 $Dir_Output_27

                    echo "Done Copy!"

                else

                    echo "Volumes less than 27, file won't be used."

                fi
                
            done
        done
    done
done