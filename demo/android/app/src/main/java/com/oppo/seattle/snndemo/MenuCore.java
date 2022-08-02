/* Copyright (C) 2020 - 2022 OPPO. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.oppo.seattle.snndemo;

import android.content.Context;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;

import java.util.ArrayList;

class MenuCore {
    private static final String TAG = "MenuCore";
    private Context mContext;
    private Menu mMenu;
    private AlgorithmConfig mAC;

    private MenuItemList mPipelineMenuItemList;
    private MenuGroupIDList mDenoiseAlgoGroupIDList;
    private MenuGroupIDList mDenoiseGroupIDList;
    private MenuGroupIDList mDenoiseAdditions;
    private MenuGroupIDList mBasicCNNGroupIDList;
    private MenuGroupIDList mClassifierAlgoGroupIDList;
    private MenuGroupIDList mClassifierAdditions;
    private MenuGroupIDList mDetectionAlgoGroupIDList;
    private MenuGroupIDList mDetectionAdditions;
    private MenuGroupIDList mStyleTransferList;

    MenuCore(Context context, Menu menu) {
        mContext = context;
        mMenu = menu;
        mAC = new AlgorithmConfig();
        init();
    }

    private void init() {
        loadPipelineItems();
        getActivePipeline();
        loadDenoiseAlgorithmItems();
        loadDenoiseItems();
        loadDenoiseAdditions();
        loadClassifierAlgorithmItems();
        loadClassifierAdditions();
        loadDetectionAlgorithmItems();
        loadDetectionAdditions();
        loadStyleTransferItems();
        setState();
    }

    private void loadStyleTransferItems() {
        mStyleTransferList = new MenuGroupIDList();
        mStyleTransferList.add(R.id.style_transfer_choices);
    }

    private void loadClassifierAlgorithmItems() {
        mClassifierAlgoGroupIDList = new MenuGroupIDList();
        mClassifierAlgoGroupIDList.add(R.id.classifiers_choices);
    }

    private void loadClassifierAdditions() {
        mClassifierAdditions = new MenuGroupIDList();
        mClassifierAdditions.add(R.id.classifier_shader_choices);
    }

    private void loadDetectionAlgorithmItems() {
        mDetectionAlgoGroupIDList = new MenuGroupIDList();
        mDetectionAlgoGroupIDList.add(R.id.detection_choices);
    }

    private void loadDetectionAdditions() {
        mDetectionAdditions = new MenuGroupIDList();
        mDetectionAdditions.add(R.id.detection_shader_choices);
    }

    private void loadPipelineItems() {
        mPipelineMenuItemList = new MenuItemList();
        mPipelineMenuItemList.add(mMenu.findItem(R.id.clearance_1_frame_pipeline));
        mPipelineMenuItemList.add(mMenu.findItem(R.id.clearance_2_frame_pipeline));
        mPipelineMenuItemList.add(mMenu.findItem(R.id.hybrid_dl_pipeline));
        mPipelineMenuItemList.add(mMenu.findItem(R.id.basic_cnn_additions));
    }

    private void getActivePipeline() {
        for (MenuItem menuItem: mPipelineMenuItemList) {
            if (menuItem.isChecked()) {
                switch (menuItem.getItemId()) {
                    case R.id.clearance_1_frame_pipeline:
                        mAC.setPipeline(AlgorithmConfig.Pipeline.CLEARANCE1FRAME);
                        break;
                    case R.id.clearance_2_frame_pipeline:
                        mAC.setPipeline(AlgorithmConfig.Pipeline.CLEARANCE2FRAME);
                        break;
                    case R.id.hybrid_dl_pipeline:
                        mAC.setPipeline(AlgorithmConfig.Pipeline.HYBRIDDL);
                        break;
                    case R.id.basic_cnn_additions:
                        mAC.setPipeline(AlgorithmConfig.Pipeline.BASIC_CNN);
                        break;
                }
            }
        }
    }

    private  void loadDenoiseAlgorithmItems() {
        mDenoiseAlgoGroupIDList = new MenuGroupIDList();
        mDenoiseAlgoGroupIDList.add(R.id.clearance_1_frame_stage_0_choices);
        mDenoiseAlgoGroupIDList.add(R.id.hybrid_dl_stage_0_choices);
    }

    private void loadDenoiseItems() {
        mDenoiseGroupIDList = new MenuGroupIDList();
        mDenoiseGroupIDList.add(R.id.clearance_1_frame_stage_1_choices);
        mDenoiseGroupIDList.add(R.id.clearance_2_frame_stage_1_choices);
        mDenoiseGroupIDList.add(R.id.hybrid_dl_stage_1_choices);
    }

    private void loadDenoiseAdditions() {
        mDenoiseAdditions = new MenuGroupIDList();
        mDenoiseAdditions.add(R.id.clearance_1_frame_stage_1_additions);
        mDenoiseAdditions.add(R.id.hybrid_dl_stage_1_additions);
    }

    boolean onOptionsItemSelected(MenuItem item) {
        item.setChecked(!item.isChecked());
        switch (item.getItemId()) {
            case R.id.hide_show_pipelines:
                hideShowPipelines(item.isChecked());
                break;
            case R.id.hide_show_stage_1:
                setDenoiserAndClearanceChoices();
                break;
            case R.id.hide_show_classifiers:
                setClassifierChoices();
                break;
            case R.id.hide_show_detectiions:
                setDetectionChoices();
                break;
            case R.id.style_transfer_choices:
                setStyleTransferChoices();
                break;
        }
        getActivePipeline();
        setDenoiserAndClearanceChoices();
        setClassifierChoices();
        setDetectionChoices();
        setStyleTransferChoices();
        setState();
        return keepMenuOpen(item);
    }

    private void hideShowPipelines(boolean b) {
        mMenu.setGroupVisible(R.id.pipeline_choices, b);
    }

    private void hideShowStyleTransfer(boolean isShow) {
        int styleTranderAlgoGroupToShow = -1;
        if (isShow) {
            styleTranderAlgoGroupToShow = R.id.style_transfer_choices;
        }
        for (Integer styleTransferAlgoGroup : mStyleTransferList) {
            mMenu.setGroupVisible(styleTranderAlgoGroupToShow, isShow && styleTransferAlgoGroup.equals(styleTranderAlgoGroupToShow));
        }
    }

    private void hideShowClassifiers(boolean isShow) {
        int classifierAlgoGroupToShow = -1;
        int classifierAdditionsToShow = -1;
        if(isShow) {
            classifierAlgoGroupToShow = R.id.classifiers_choices;
        }
        if(!mMenu.findItem(R.id.no_classifier).isChecked()) {
            classifierAdditionsToShow = R.id.classifier_shader_choices;
        }
        for(Integer classifierAlgoGroup : mClassifierAlgoGroupIDList) {
            mMenu.setGroupVisible(classifierAlgoGroup, isShow & classifierAlgoGroup.equals(classifierAlgoGroupToShow));
        }
        for(Integer classifierAddition : mClassifierAdditions) {
            mMenu.setGroupVisible(classifierAddition, isShow & classifierAddition.equals(classifierAdditionsToShow));
        }
    }

    private void hideShowDetections(boolean isShow) {
        int detectionAlgoGroupToShow = -1;
        int detectionAdditionsToShow = -1;
        if(isShow) {
            detectionAlgoGroupToShow = R.id.detection_choices;
        }
        if(!mMenu.findItem(R.id.no_detection).isChecked()) {
            detectionAdditionsToShow = R.id.detection_shader_choices;
        }
        for(Integer detectionAlgoGroup : mDetectionAlgoGroupIDList) {
            mMenu.setGroupVisible(detectionAlgoGroup, isShow && detectionAlgoGroup.equals(detectionAlgoGroupToShow));
        }
        for(Integer detectionAddition : mDetectionAdditions) {
            mMenu.setGroupVisible(detectionAddition, isShow && detectionAddition.equals(detectionAdditionsToShow));
        }
    }

    private void setClassifierChoices() {
        MenuItem hideShowClassifiers = mMenu.findItem(R.id.hide_show_classifiers);
        hideShowClassifiers(hideShowClassifiers.isChecked());
    }

    private void setDetectionChoices() {
        MenuItem hideShowDetections = mMenu.findItem(R.id.hide_show_detectiions);
        hideShowDetections(hideShowDetections.isChecked());
    }

    private void setDenoiserAndClearanceChoices() {
        MenuItem hideShowDenoisers = mMenu.findItem(R.id.hide_show_stage_1);
        hideShowDenoisers(hideShowDenoisers.isChecked());
    }

    private void setStyleTransferChoices() {
        MenuItem hideShowStyle = mMenu.findItem(R.id.style_transfer);
        hideShowStyleTransfer(hideShowStyle.isChecked());
    }

    private void hideShowDenoisers(boolean denoisers) {
        int denoiserAlgoGroupToshow = -1;
        int denoiserGroupToShow = -1;
        int denoiserAdditionsToShow = -1;
        switch (mAC.getPipeline()) {
            case CLEARANCE1FRAME:
                denoiserAlgoGroupToshow = R.id.clearance_1_frame_stage_0_choices;
                if (!mMenu.findItem(R.id.clearance_1_frame_no_denoise).isChecked()) {
                    denoiserGroupToShow = R.id.clearance_1_frame_stage_1_choices;
                    denoiserAdditionsToShow = R.id.clearance_1_frame_stage_1_additions;
                }
                break;
            case CLEARANCE2FRAME:
                denoiserGroupToShow = R.id.clearance_2_frame_stage_1_choices;
                break;
            case HYBRIDDL:
                denoiserAlgoGroupToshow = R.id.hybrid_dl_stage_0_choices;
                if (!mMenu.findItem(R.id.hybrid_dl_no_denoiser).isChecked()) {
                    denoiserGroupToShow = R.id.hybrid_dl_stage_1_choices;
                    denoiserAdditionsToShow = R.id.hybrid_dl_stage_1_additions;
                }
                break;
        }
        for (Integer denoiseAlgoGroup : mDenoiseAlgoGroupIDList) {
            mMenu.setGroupVisible(denoiseAlgoGroup, denoisers && denoiseAlgoGroup.equals(denoiserAlgoGroupToshow));
        }
        for (Integer denoiseGroup : mDenoiseGroupIDList) {
            mMenu.setGroupVisible(denoiseGroup, denoisers && denoiseGroup.equals(denoiserGroupToShow));
        }
        for (Integer denoiseAddition : mDenoiseAdditions) {
            mMenu.setGroupVisible(denoiseAddition, denoisers && denoiseAddition.equals(denoiserAdditionsToShow));
        }
    }

    private void setState() {
        switch (mAC.getPipeline()) {
            case CLEARANCE1FRAME:
                if (mMenu.findItem( R.id.clearance_1_frame_no_denoise).isChecked()) mAC.setDenoiserAlgorithm(AlgorithmConfig.DenoiserAlgorithm.NONE);
                if (mMenu.findItem( R.id.clearance_compute_shader).isChecked()) mAC.setDenoiser(AlgorithmConfig.Denoiser.COMPUTESHADER);
                if (mMenu.findItem( R.id.clearance_fragment_shader).isChecked()) mAC.setDenoiser(AlgorithmConfig.Denoiser.FRAGMENTSHADER);
                if (mMenu.findItem( R.id.clearance_1_frame_aidenoise).isChecked()) mAC.setDenoiserAlgorithm(AlgorithmConfig.DenoiserAlgorithm.AIDENOISER);
                if (mMenu.findItem( R.id.clearance_1_frame_spatialdenoise).isChecked()) mAC.setDenoiserAlgorithm(AlgorithmConfig.DenoiserAlgorithm.SPATIALDENOISER);

                if (!mMenu.findItem( R.id.clearance_1_frame_no_denoise).isChecked())
                    if (mMenu.findItem(R.id.clearance_toggle_ssbo).isChecked()) mAC.setSSBO(true);
                    else mAC.setSSBO(false);

                if(mMenu.findItem(R.id.no_classifier).isChecked()) mAC.setClassifierAlgorithm(AlgorithmConfig.ClassifierAlgorithm.NONE);
                if(mMenu.findItem(R.id.resnet18_classifier).isChecked()) mAC.setClassifierAlgorithm(AlgorithmConfig.ClassifierAlgorithm.RESNET18);
                if(mMenu.findItem(R.id.mobilenetv2_classifier).isChecked()) mAC.setClassifierAlgorithm(AlgorithmConfig.ClassifierAlgorithm.MOBILENETV2);
                if(mMenu.findItem(R.id.classifier_fragment_shader).isChecked()) mAC.setClassifier(AlgorithmConfig.Classifier.FRAGMENTSHADER);
                if(mMenu.findItem(R.id.classifier_compute_shader).isChecked()) mAC.setClassifier(AlgorithmConfig.Classifier.COMPUTESHADER);

                if(mMenu.findItem(R.id.no_detection).isChecked()) mAC.setDetectionAlgorithm(AlgorithmConfig.DetectionAlgorithm.NONE);
                if(mMenu.findItem(R.id.yolov3_detection).isChecked()) mAC.setDetectionAlgorithm(AlgorithmConfig.DetectionAlgorithm.YOLOV3);
                if(mMenu.findItem(R.id.detection_fragment_shader).isChecked()) mAC.setDetection(AlgorithmConfig.Detection.FRAGMENTSHADER);
                if(mMenu.findItem(R.id.detection_compute_shader).isChecked()) mAC.setDetection(AlgorithmConfig.Detection.COMPUTESHADER);

                if(mMenu.findItem(R.id.no_style).isChecked()) mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.NONE);
                if(mMenu.findItem(R.id.style_candy).isChecked()) mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.CANDY);
                if(mMenu.findItem(R.id.style_mosaic).isChecked()) mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.MOSAIC);
                if(mMenu.findItem(R.id.style_pointilism).isChecked()) mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.POINTILISM);
                if(mMenu.findItem(R.id.style_rain_princess).isChecked()) mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.RAIN_PRINCESS);
                if(mMenu.findItem(R.id.style_udnie).isChecked()) mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.UDNIE);

                break;

            case CLEARANCE2FRAME:
                if (mMenu.findItem( R.id.clearance_2_frame_no_denoise).isChecked()) mAC.setDenoiserAlgorithm(AlgorithmConfig.DenoiserAlgorithm.NONE);

                break;

            case HYBRIDDL:
                if (mMenu.findItem( R.id.hybrid_dl_no_denoiser).isChecked()) mAC.setDenoiserAlgorithm(AlgorithmConfig.DenoiserAlgorithm.NONE);
                if (mMenu.findItem( R.id.hybrid_dl_compute_shader).isChecked()) mAC.setDenoiser(AlgorithmConfig.Denoiser.COMPUTESHADER);
                if (mMenu.findItem( R.id.hybrid_dl_fragment_shader).isChecked()) mAC.setDenoiser(AlgorithmConfig.Denoiser.FRAGMENTSHADER);
                if (mMenu.findItem( R.id.clearance_1_frame_aidenoise).isChecked()) mAC.setDenoiserAlgorithm(AlgorithmConfig.DenoiserAlgorithm.AIDENOISER);
                if (mMenu.findItem( R.id.clearance_1_frame_spatialdenoise).isChecked()) mAC.setDenoiserAlgorithm(AlgorithmConfig.DenoiserAlgorithm.SPATIALDENOISER);

                if (!mMenu.findItem( R.id.hybrid_dl_no_denoiser).isChecked())
                    if (mMenu.findItem(R.id.hybrid_dl_toggle_ssbo).isChecked()) mAC.setSSBO(true);
                    else mAC.setSSBO(false);

                mAC.setTemporalFilter(false);
                break;

            case BASIC_CNN:

                if (mMenu.findItem(R.id.basic_cnn_additions).isChecked()) mAC.setBasicCNN(AlgorithmConfig.BasicCNN.BASIC_CNN);
                break;
        }
    }

    public AlgorithmConfig getState() {
        return mAC;
    }

    private boolean keepMenuOpen(MenuItem item) {
        item.setShowAsAction(MenuItem.SHOW_AS_ACTION_COLLAPSE_ACTION_VIEW);
        item.setActionView(new View(mContext));
        item.setOnActionExpandListener(new MenuItem.OnActionExpandListener(){
            @Override
            public boolean onMenuItemActionExpand(MenuItem item){
                return false;
            }

            @Override
            public boolean onMenuItemActionCollapse(MenuItem item){
                return false;
            }
        });
        return false;
    }

    private class MenuItemList extends ArrayList<MenuItem>{

    }

    private class MenuGroupIDList extends ArrayList<Integer>{

    }
}
