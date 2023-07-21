SCARED STTN for dimiP:
-----------------------------
python train.py 
--shifted 
--config configs/hyper-kvasir-hoover-SCARED.json 
--model sttn 
-i release_model/sttn_hyper-kvasir-transfer2-TransMask/

nohup python train.py --shifted --config configs/hyper-kvasir-hoover-SCARED.json --model sttn -i release_model/sttn_hyper-kvasir-transfer2-TransMask/ > nohupSCARED.out &

- gen 181
python test.py  --gpu 0 --overlaid --output release_model/sttn_hyper-kvasir-hoover-SCARED/ --frame /home/datasets/SCARED/ds1/JPEGImages/kf1/ --mask /home/datasets/SCARED/ds1/Annotations/kf1/ --ckptpath release_model/sttn_hyper-kvasir-hoover-SCARED/ --ckptnumber 4501
python test.py  --gpu 0 --overlaid --output release_model/sttn_hyper-kvasir-hoover-SCARED/ --frame /home/datasets/SCARED/ds1/JPEGImages/kf2/ --mask /home/datasets/SCARED/ds1/Annotations/kf2/ --ckptpath release_model/sttn_hyper-kvasir-hoover-SCARED/ --ckptnumber 4501
python test.py  --gpu 0 --overlaid --output release_model/sttn_hyper-kvasir-hoover-SCARED/ --frame /home/datasets/SCARED/ds1/JPEGImages/kf3/ --mask /home/datasets/SCARED/ds1/Annotations/kf3/ --ckptpath release_model/sttn_hyper-kvasir-hoover-SCARED/ --ckptnumber 4501

---------------

%% train by initializing with Model_{S,R}
python train.py 
--shifted 
--config configs/hyper-kvasir-hoover-SCARED-InitSR-500freq.json 
--model sttn 
-i release_model/sttn_hyper-kvasir/

python train.py --shifted --config configs/hyper-kvasir-hoover-SCARED-InitSR-500freq.json --model sttn -i release_model/sttn_hyper-kvasir/
python test.py  --gpu 0 --overlaid --output release_model/sttn_hyper-kvasir-hoover-SCARED-InitSR-500freq/ --frame /home/datasets/SCARED/ds1/JPEGImages/kf1/ --mask /home/datasets/SCARED/ds1/Annotations/kf1/ --ckptpath release_model/sttn_hyper-kvasir-hoover-SCARED-InitSR-500freq/ --ckptnumber 81

python train.py --shifted --config configs/hyper-kvasir-hoover-SCARED-InitSR-500freq-8vid.json --model sttn -i release_model/sttn_hyper-kvasir/

%% train from scratch
python train.py --shifted --config configs/hyper-kvasir-hoover-SCARED.json --model sttn
python test.py  --gpu 0 --overlaid --output release_model/sttn_hyper-kvasir-hoover-SCARED/ --frame /home/datasets/SCARED/ds1/JPEGImages/kf1/ --mask /home/datasets/SCARED/ds1/Annotations/kf1/ --ckptpath release_model/sttn_hyper-kvasir-hoover-SCARED/ --ckptnumber 81


%% train by initializing with Model_{S,R} %
%% Change code to take in random masks instead of loaded. 
%% then train 
python train.py 
--shifted 
--config configs/hyper-kvasir-hoover-SCARED-InitSR-500freq-RandMasks.json 
--model sttn 
-i release_model/sttn_hyper-kvasir/

%% test with Model_{S,R} directly


#################
generating inpainted dataset
nohup python test.py  --gpu 0 --overlaid \
--output /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/JPEGImagesNSfoldersOriginal/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/JPEGImages/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/Annotations/ \
-c /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-transfer2-TransMask \
-cn 9 \
--zip > generatinginpaintingfulltest.out&

# as a sanity check
python test.py  --gpu "0,1,2" --overlaid \
--output /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/JPEGImagesNStesting/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ \
-c release_model/sttn_hyper-kvasir-transfer2-TransMask \
-cn 9 \

train not shifted
nohup python train.py --config configs/hyper-kvasir-Addingspec-GTdata.json --model sttn > hyper-kvasir-Addingspec-GTdata.out &
python test.py  --gpu 6 --nomask \
--output release_model/sttn_hyper-kvasir-Addingspec-GTdata/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ \
-c release_model/sttn_hyper-kvasir-Addingspec-GTdata \
-cn 12 \

try to save until 20000 iterations and if not good then train using pretrained model:
nohup python train.py --config configs/hyper-kvasir-Addingspec-GTdata-pretrained.json --model sttn -i release_model/sttn_hyper-kvasir-transfer2-TransMask > hyper-kvasir-Addingspec-GTdata-pretrained.out &

#####clara data 21-2-2023
nohup python test.py  --gpu 1 --overlaid \
--output /home/datasets/ClaraData/Inpainted/ \
--frame /home/datasets/ClaraData/JPEGImages/ \
--mask /home/datasets/ClaraData/Annotations/ \
-c release_model/sttn_hyper-kvasir-transfer2-TransMask \
-cn 9 \
--zip > ClaraData.out&



###debugging training no mask add spec results
nohup python train.py --config configs/hyper-kvasir-Addingspec-GTdata.json --model sttn > hyper-kvasir-Addingspec-GTdata.out &

python test.py  --gpu 1 \
--output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ \
-c /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata/ \
-cn 8 \
--nomask

python test.py  --gpu 0 \
--output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/testFrames/hyperK_372/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/testAnnot/hyperK_372/ \
-c /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata/ \
-cn 8 \
--nomask

python test.py  --gpu 0 \
--output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-transfer2-TransMask/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/testFrames/hyperK_372/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/testAnnot/hyperK_372/ \
-c /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-transfer2-TransMask/ \
-cn 9


# date today: 9-6-2023
nohup python train.py --config /home/datasets/Trainedmodelsandresults/Endo-STTN/configs/hyper-kvasir-Addingspec-GTdata-samplefrom20.json --model sttn > hyper-kvasir-Addingspec-GTdata-samplefrom20.out &
python test_strideref.py  --gpu 4 -s 20 -r 0 --nomask --output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ --mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ --ckptpath /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --ckptnumber 3
python test_strideref.py  --gpu 4 -s 20 -r 0 --nomask --output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ --mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ --ckptpath /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --ckptnumber 8
python test_strideref.py  --gpu 4 -s 20 -r 0 --nomask --output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ --mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ --ckptpath /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --ckptnumber 9

python test_strideref.py  --gpu 4 -s 5 -r 4 --nomask --output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ --mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ --ckptpath /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --ckptnumber 3
python test_strideref.py  --gpu 4 -s 5 -r 4 --nomask --output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ --mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ --ckptpath /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --ckptnumber 8
python test_strideref.py  --gpu 4 -s 5 -r 4 --nomask --output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ --mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ --ckptpath /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --ckptnumber 9

python test_strideref.py  --gpu 4 -s 100 -r 0 --nomask --output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ --mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ --ckptpath /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --ckptnumber 3
python test_strideref.py  --gpu 4 -s 100 -r 0 --nomask --output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ --mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ --ckptpath /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --ckptnumber 8
python test_strideref.py  --gpu 4 -s 100 -r 0 --nomask --output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ --mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ --ckptpath /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --ckptnumber 9

python test_strideref.py  --gpu 4 -s 5 -r 10 --nomask --output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ --mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ --ckptpath /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --ckptnumber 3
python test_strideref.py  --gpu 4 -s 5 -r 10 --nomask --output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ --mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ --ckptpath /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --ckptnumber 8
python test_strideref.py  --gpu 4 -s 5 -r 10 --nomask --output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ --mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ --ckptpath /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-samplefrom20/ --ckptnumber 9


python test.py  --gpu 4 --nomask \
--output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ \
-c /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata/ \
-cn 3 \

#masked addition:
nohup python train.py --config /home/datasets/Trainedmodelsandresults/Endo-STTN/configs/hyper-kvasir-Addingspec-GTdata-masked.json --model sttn > hyper-kvasir-Addingspec-GTdata-masked.out &

python test.py  --gpu 4 --overlaid --oppmask \
--output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-masked/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ \
-c /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-masked/ \
-cn 3

python test.py  --gpu 4 --overlaid --oppmask \
--output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-masked/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ \
-c /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-masked/ \
-cn 8

python test.py  --gpu 4 --overlaid --Dil 0 --oppmask \
--output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-masked/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ \
-c /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-masked/ \
-cn 3

python test.py  --gpu 4 --overlaid --Dil 0 --oppmask \
--output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-masked/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ \
-c /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-masked/ \
-cn 8

python test.py  --gpu 4 --oppmask \
--output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-masked/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ \
-c /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-masked/ \
-cn 3

python test.py  --gpu 4 --oppmask \
--output /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-masked/ \
--frame /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testFrames/hyperK_343/ \
--mask /home/datasets/Hyper_Kvasir/Resized_Renamed_Images_withMasks/Data_NotCropped_usedForTrainingEndoSTTNoriginally/testAnnot/hyperK_343/ \
-c /home/datasets/Trainedmodelsandresults/Endo-STTN/release_model/sttn_hyper-kvasir-Addingspec-GTdata-masked/ \
-cn 8
