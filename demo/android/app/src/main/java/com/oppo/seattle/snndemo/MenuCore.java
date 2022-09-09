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

    private MenuGroupIDList mDenoiseAlgoGroupIDList;
    private MenuGroupIDList mDenoiseGroupIDList;
    private MenuGroupIDList mDenoiseAdditions;
    private MenuGroupIDList mClassifierAlgoGroupIDList;
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
    }

    private void loadDetectionAlgorithmItems() {
        mDetectionAlgoGroupIDList = new MenuGroupIDList();
        mDetectionAlgoGroupIDList.add(R.id.detection_choices);
    }

    private void loadDetectionAdditions() {
        mDetectionAdditions = new MenuGroupIDList();
        mDetectionAdditions.add(R.id.detection_shader_choices);
    }

    private void getActivePipeline() {
    }

    private  void loadDenoiseAlgorithmItems() {
        mDenoiseAlgoGroupIDList = new MenuGroupIDList();
        mDenoiseAlgoGroupIDList.add(R.id.clearance_1_frame_stage_0_choices);
    }

    private void loadDenoiseItems() {
        mDenoiseGroupIDList = new MenuGroupIDList();
    }

    private void loadDenoiseAdditions() {
        mDenoiseAdditions = new MenuGroupIDList();
    }

    boolean onOptionsItemSelected(MenuItem item) {
        item.setChecked(!item.isChecked());
        switch (item.getItemId()) {
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
        if(isShow) {
            classifierAlgoGroupToShow = R.id.classifiers_choices;
        }
        for(Integer classifierAlgoGroup : mClassifierAlgoGroupIDList) {
            mMenu.setGroupVisible(classifierAlgoGroup, isShow & classifierAlgoGroup.equals(classifierAlgoGroupToShow));
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
                if (mMenu.findItem( R.id.clearance_1_frame_spatialdenoise).isChecked()) mAC.setDenoiserAlgorithm(AlgorithmConfig.DenoiserAlgorithm.SPATIALDENOISER);

                if (!mMenu.findItem( R.id.clearance_1_frame_no_denoise).isChecked())
                   mAC.setSSBO(false);

                if(mMenu.findItem(R.id.no_classifier).isChecked()) mAC.setClassifierAlgorithm(AlgorithmConfig.ClassifierAlgorithm.NONE);
                if(mMenu.findItem(R.id.resnet18_classifier).isChecked()) mAC.setClassifierAlgorithm(AlgorithmConfig.ClassifierAlgorithm.RESNET18);
                if(mMenu.findItem(R.id.mobilenetv2_classifier).isChecked()) mAC.setClassifierAlgorithm(AlgorithmConfig.ClassifierAlgorithm.MOBILENETV2);

                if(mMenu.findItem(R.id.no_detection).isChecked()) mAC.setDetectionAlgorithm(AlgorithmConfig.DetectionAlgorithm.NONE);
                if(mMenu.findItem(R.id.yolov3_detection).isChecked()) mAC.setDetectionAlgorithm(AlgorithmConfig.DetectionAlgorithm.YOLOV3);

                if(mMenu.findItem(R.id.no_style).isChecked()) mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.NONE);
                if(mMenu.findItem(R.id.style_candy).isChecked()) mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.CANDY);
                if(mMenu.findItem(R.id.style_mosaic).isChecked()) mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.MOSAIC);
                if(mMenu.findItem(R.id.style_pointilism).isChecked()) mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.POINTILISM);
                if(mMenu.findItem(R.id.style_rain_princess).isChecked()) mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.RAIN_PRINCESS);
                if(mMenu.findItem(R.id.style_udnie).isChecked()) mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.UDNIE);
                break;
            case HYBRIDDL:
                if (mMenu.findItem( R.id.clearance_1_frame_spatialdenoise).isChecked()) mAC.setDenoiserAlgorithm(AlgorithmConfig.DenoiserAlgorithm.SPATIALDENOISER);
                mAC.setTemporalFilter(false);
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

    private class MenuGroupIDList extends ArrayList<Integer>{

    }
}
