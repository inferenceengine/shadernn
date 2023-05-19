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

class MenuCore {
    private static final String TAG = "MenuCore";
    private Context mContext;
    private Menu mMenu;
    private AlgorithmConfig mAC;

    MenuCore(Context context, Menu menu) {
        mContext = context;
        mMenu = menu;
        mAC = new AlgorithmConfig();
    }

    boolean onOptionsItemSelected(MenuItem item) {
        item.setChecked(!item.isChecked());
        if (item.getItemId() == R.id.fp16 || item.getItemId() == R.id.fp32
            || item.getItemId() == R.id.detection_compute_shader || item.getItemId() == R.id.detection_fragment_shader) {
            return keepMenuOpen(item);
        }
        setState();
        return true;
    }

    private void setState() {
        boolean computeShader = mMenu.findItem( R.id.detection_compute_shader).isChecked();
        if (mMenu.findItem(R.id.spatialdenoise).isChecked()) {
            mAC.setDenoiserAlgorithm(AlgorithmConfig.DenoiserAlgorithm.SPATIALDENOISER);
            if (computeShader) {
                mAC.setDenoiser(AlgorithmConfig.DenoiserShader.COMPUTESHADER);
            } else {
                mAC.setDenoiser(AlgorithmConfig.DenoiserShader.FRAGMENTSHADER);
            }
        } else if (mMenu.findItem(R.id.resnet18_classifier).isChecked()) {
            mAC.setClassifierAlgorithm(AlgorithmConfig.ClassifierAlgorithm.RESNET18);
            if (computeShader) {
                mAC.setClassifier(AlgorithmConfig.ClassifierShader.COMPUTESHADER);
            } else {
                mAC.setClassifier(AlgorithmConfig.ClassifierShader.FRAGMENTSHADER);
            }
        } else if (mMenu.findItem(R.id.mobilenetv2_classifier).isChecked()) {
            mAC.setClassifierAlgorithm(AlgorithmConfig.ClassifierAlgorithm.MOBILENETV2);
            if (computeShader) {
                mAC.setClassifier(AlgorithmConfig.ClassifierShader.COMPUTESHADER);
            } else {
                mAC.setClassifier(AlgorithmConfig.ClassifierShader.FRAGMENTSHADER);
            }
        } else if (mMenu.findItem(R.id.yolov3_detection).isChecked()) {
            mAC.setDetectionAlgorithm(AlgorithmConfig.DetectionAlgorithm.YOLOV3);
            if (computeShader) {
                mAC.setDetection(AlgorithmConfig.DetectionShader.COMPUTESHADER);
            } else {
                mAC.setDetection(AlgorithmConfig.DetectionShader.FRAGMENTSHADER);
            }
        } else if (mMenu.findItem(R.id.style_candy).isChecked()) {
            mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.CANDY);
        } else if (mMenu.findItem(R.id.style_mosaic).isChecked()) {
            mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.MOSAIC);
        } else if (mMenu.findItem(R.id.style_pointilism).isChecked()) {
            mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.POINTILISM);
        } else if (mMenu.findItem(R.id.style_rain_princess).isChecked()) {
            mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.RAIN_PRINCESS);
        } else if (mMenu.findItem(R.id.style_udnie).isChecked()) {
            mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.UDNIE);
        }
        if (mMenu.findItem( R.id.fp32).isChecked()) {
            mAC.setPrecision(AlgorithmConfig.Precision.FP32);
        } else {
            mAC.setPrecision(AlgorithmConfig.Precision.FP16);
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
}
