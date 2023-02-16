#!/usr/bin/env bash

### Use above for the debug with bashdb to run properly
### Adapted from the script writted by Xiaoxiao Wang (Wang et al., 2019)
### Author: Han Wang
### 06-Feb-2023: The script now outputs the training (18 subj), validation (2 subj), and testing datasets (5 subj).
### 30-Jan-2023: Changing the script to extract images per task per block and output all files to one folder.

### basic setups
#### folders for all participants' preprocessed data of the dual-task.
Dir_Common="/mnt/g/Backup/fMRI_dualtask/1_processed/batch_processing_spmeditor/Data"

Dir_TMP="$Dir_Common/CNN/Output_Extracted/tmp" # a temporary folder
Dir_Output="$Dir_Common/CNN/Output_Extracted/" # folder for output (31 volumes maximum, NOT yet extracted for training, etc.)

Dat_type_txt=$Dir_TMP/train_va_test.txt
tr=`sed -n "2,1p" $Dat_type_txt|awk '{print $2}'`
va=`sed -n "8,1p" $Dat_type_txt|awk '{print $2}'`
te=`sed -n "3,1p" $Dat_type_txt|awk '{print $2}'`


Dir_Train="$Dir_Common/CNN/Output_Extracted/dat_train" # folder for training dataset output (random 27 volumes)
Dir_Validate="$Dir_Common/CNN/Output_Extracted/dat_validate" # folder for validation dataset output (first 27 volumes)
Dir_Test="$Dir_Common/CNN/Output_Extracted/dat_test" # folder for testing dataset output (first 27 volumnes)


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

        for ((iCond=1; iCond<=$Num_Cond; iCond++ ))
        do
            for ((iRepCond=1; iRepCond<=$Num_RepCond; iRepCond++ ))
            do
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

                # Get the index of the last volume
                Ind_Last=`3dinfo -n4 $File_InputNii`
                Ind_Last=`echo $Ind_Last | awk '{print $NF}'`
                Ind_Last=$(echo "$Ind_Last-1" |bc)
                
                # Get the volumn indices
                Ind_TaskBegin=$(echo "scale=0;$Dig_TaskBegin_Sec/$Dig_TR_Sec" |bc)
                Ind_TaskEnd=$(echo "$Ind_TaskBegin+$Num_DurTask_Ind-1" |bc)

                if [[ $Ind_TaskEnd -gt $Ind_Last ]]
                then
                    Ind_TaskEnd=$Ind_Last
                    Num_DurTask_Ind=$(echo "$Ind_TaskEnd-$Ind_TaskBegin+1" |bc)
                fi

                # Extract the volumes
                echo "Phase 1: Extracting volumes for $Subj_tmp Run ${iRun} Cond ${iCond} RepCond ${iRepCond}..."
                File_OutputMax31=$Dir_Output/$Subj_tmp'_Run-'$(echo "$iRun" |bc)'_Cond-'$(echo "$iCond" |bc)'_RepCond-'$(echo "$iRepCond" |bc)'_Vol-'$(echo "$Num_DurTask_Ind" |bc).nii.gz
                3dTcat -prefix $File_OutputMax31 \
                        $File_InputNii'['$Ind_TaskBegin'..'$Ind_TaskEnd']'
                echo "Done Segmentation 1!"


                # Further extract 27 vols for training, validation, and testing datasets

                Dat_type=`sed -n "$(echo "$iSubj+1" |bc),1p" $Dat_type_txt|awk '{print $2}'`
                echo "Dat_type is $Dat_type"

                if [[ $Num_DurTask_Ind -gt 26 ]]
                then
                    if [ "$Dat_type" = "$te" ] || [ "$Dat_type" = "$va" ]
                    then
                        echo "Will get the first 27 volumes!"
                        Ind_SegBegin=0
                        Ind_SegEnd=26
                        DiffIndSeg=$(echo "$Ind_SegEnd-$Ind_SegBegin+1" |bc)
                    else
                        echo "Will randomly select 27 volumes!"
                        Ind_SegBegin=9999
                        Ind_SegEnd=9999
                        DiffIndSeg=$(echo "$Ind_SegEnd-$Ind_SegBegin+1" |bc)

                        while [ $DiffIndSeg != 27 ]
                        do
                            Ind_Seg=`shuf -i 0-$(echo "$Num_DurTask_Ind-1" |bc) -n 2`
                            sortedInd_Seg=( $( printf "%s\n" "${Ind_Seg[@]}" | sort -n ) )
                            #Ind_SegBegin=`echo $Ind_Seg | while read -r c1 c2; do echo $c1; done`
                            Ind_SegBegin=${sortedInd_Seg[0]}
                            #echo "Ind_SegBegin $Ind_SegBegin"
                            #Ind_SegEnd=`echo $Ind_Seg | while read -r c1 c2; do echo $c2; done`
                            Ind_SegEnd=${sortedInd_Seg[1]}
                            #echo "Ind_SegEnd $Ind_SegEnd"
                            DiffIndSeg=$(echo "$Ind_SegEnd-$Ind_SegBegin+1" |bc)
                            #echo "DiffIndSeg $DiffIndSeg"
                            #DiffIndSeg=${DiffIndSeg#-}
                            #echo "ABS DiffIndSeg $DiffIndSeg"
                        done
                    fi


                    echo "Phase 2: Extracting 27 volumes as final output..."

                    if [[ "$Dat_type" = "$te" ]]
                    then 
                        Dir_Output_27=$Dir_Test
                    elif [[ "$Dat_type" = "$va" ]]
                    then
                        Dir_Output_27=$Dir_Validate
                    else
                        Dir_Output_27=$Dir_Train
                    fi


                    File_Output27=$Dir_Output_27/$Subj_tmp'_Run-'$(echo "$iRun" |bc)'_Cond-'$(echo "$iCond" |bc)'_RepCond-'$(echo "$iRepCond" |bc)'_Vol-'$(echo "$Ind_SegBegin" |bc)'-'$(echo "$Ind_SegEnd" |bc)'-'$(echo "$DiffIndSeg" |bc).nii.gz
                    3dTcat -prefix $File_Output27 \
                            $File_OutputMax31'['$Ind_SegBegin'..'$Ind_SegEnd']'
                    echo "Done Segmentation 2!"
                fi
            done
        done
    done
done