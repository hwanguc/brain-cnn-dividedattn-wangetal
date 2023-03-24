# brain-conv-dividedattn-wangetal

This respiratory contains the scripts for a transfer learning pipeline for a 3D convolutional network that 
predicts fMRI task state under a divided-attention speech and visual dual task.

## processing_pipelines/step0_labeldata

**step0_1_getInput_resample.sh** - a pipeline that segments and resamples the input image data.

## processing_pipelines/step1_transfer_learning

**main.py**: main script for the transfer learning pipeline.

**dataset.py**: read and normalise the preprocessed data.

**get_pretrained_model**: load the pretrained model.

**load_save_checkpoint.py**: load or save the model state.

**model_o.py**: a 3D convolutional network model proposed by Wang et al. (2019).

**train.py**: model training.

**test.py**: model testing.
