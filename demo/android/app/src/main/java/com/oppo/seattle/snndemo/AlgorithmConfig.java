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

public class AlgorithmConfig {

    public enum DenoiserAlgorithm {
        NONE, AIDENOISER, SPATIALDENOISER
    }

    public enum DenoiserShader {
        COMPUTESHADER, FRAGMENTSHADER
    }

    public enum ClassifierAlgorithm {
        NONE, RESNET18, MOBILENETV2
    }

    public enum ClassifierShader {
        COMPUTESHADER, FRAGMENTSHADER
    }

    public enum DetectionAlgorithm {
        NONE, YOLOV3
    }

    public enum DetectionShader {
        COMPUTESHADER, FRAGMENTSHADER
    }

    public enum Precision {
        FP32, FP16
    }

    public enum StyleTransfer {
        NONE, CANDY, MOSAIC, POINTILISM, RAIN_PRINCESS, UDNIE
    }

    public int classifierIndex;
    private String[] resnet18Classes = {"None", "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    private String[] mobilenetClasses = {"None", "Class 1", "Class 2"};

    private DenoiserShader denoiserShader;
    private DenoiserAlgorithm denoiserAlgorithm;
    private ClassifierAlgorithm classifierAlgorithm;
    private ClassifierShader classifierShader;
    private DetectionAlgorithm detectionAlgorithm;
    private DetectionShader detectionShader;
    private Precision precision;
    private boolean temporalFilter;
    private StyleTransfer styleTransferAlgorithm;

    AlgorithmConfig() {
        init();
    }

    private void init() {
        denoiserAlgorithm = DenoiserAlgorithm.NONE;
        denoiserShader = DenoiserShader.FRAGMENTSHADER;
        classifierAlgorithm = ClassifierAlgorithm.NONE;
        classifierShader = ClassifierShader.FRAGMENTSHADER;
        detectionAlgorithm = DetectionAlgorithm.NONE;
        detectionShader = DetectionShader.FRAGMENTSHADER;
        precision = Precision.FP32;
        styleTransferAlgorithm = StyleTransfer.NONE;
        temporalFilter = false;
    }

    public String getClassifierOutput() {
        String classifierOutput = "";
        if (this.classifierAlgorithm == ClassifierAlgorithm.RESNET18) {
            classifierOutput = (classifierIndex >= 0 && classifierIndex <= 10) ? resnet18Classes[classifierIndex] : "None";
        } else if (this.classifierAlgorithm == ClassifierAlgorithm.MOBILENETV2) {
            classifierOutput = (classifierIndex >= 0 && classifierIndex <= 2) ? mobilenetClasses[classifierIndex] : "None";
        } else {
            classifierOutput = "N/A";
        }

        return classifierOutput;
    }

    void setDenoiserAlgorithm(DenoiserAlgorithm denoiserAlgorithm) {
        this.denoiserAlgorithm = denoiserAlgorithm;
    }

    public void setDenoiser(DenoiserShader denoiserShader) {
        this.denoiserShader = denoiserShader;
    }

    public boolean isDenoiseSPATIALDENOISER() {return  denoiserAlgorithm == DenoiserAlgorithm.SPATIALDENOISER; }

    public boolean isDenoiseComputeShader() {
        return denoiserShader == DenoiserShader.COMPUTESHADER;
    }

    void setTemporalFilter(boolean temporalFilter) {
        this.temporalFilter = temporalFilter;
    }

    void setStyleTransferAlgorithm(StyleTransfer styleTransferAlgorithm) {
        this.styleTransferAlgorithm = styleTransferAlgorithm;
    }
    public boolean isStyleTransferNONE() { return styleTransferAlgorithm == StyleTransfer.NONE; }

    public boolean isStyleTransferCANDY() { return styleTransferAlgorithm == StyleTransfer.CANDY; }

    public boolean isStyleTransferMOSAIC() { return styleTransferAlgorithm == StyleTransfer.MOSAIC; }

    public boolean isStyleTransferPOINTILISM() { return styleTransferAlgorithm == StyleTransfer.POINTILISM; }

    public boolean isStyleTransferRAINPRINCESS() { return styleTransferAlgorithm == StyleTransfer.RAIN_PRINCESS; }

    public boolean isStyleTransferUDNIE() { return  styleTransferAlgorithm == StyleTransfer.UDNIE; }

    public void setClassifierAlgorithm(ClassifierAlgorithm classifierAlgorithm) {
        this.classifierAlgorithm = classifierAlgorithm;
        this.classifierIndex = 0;
    }

    public void setClassifier(ClassifierShader classifierShader) {
        this.classifierShader = classifierShader;
        this.classifierIndex = 0;
    }

    public boolean isClassifierNONE() {
        return this.classifierAlgorithm == ClassifierAlgorithm.NONE;
    }

    public boolean isClassifierResnet18() {
        return this.classifierAlgorithm == ClassifierAlgorithm.RESNET18;
    }

    public boolean isClassifierMobilenetv2() {
        return this.classifierAlgorithm == ClassifierAlgorithm.MOBILENETV2;
    }

    public boolean isClassifierComputeShader() {
        return this.classifierShader == ClassifierShader.COMPUTESHADER;
    }

    public void setDetectionAlgorithm(DetectionAlgorithm detectionAlgorithm) {
        this.detectionAlgorithm = detectionAlgorithm;
    }

    public void setDetection(DetectionShader detectionShader) {
        this.detectionShader = detectionShader;
    }

    public void setPrecision(Precision precision) {
        this.precision = precision;
    }

    public boolean isDetectionYolov3() {
        return this.detectionAlgorithm == DetectionAlgorithm.YOLOV3;
    }

    public boolean isDetectionComputeShader() {
        return this.detectionShader == DetectionShader.COMPUTESHADER;
    }

    public Precision getPrecision() {
        return precision;
    }

    public boolean isFP16() {
        return this.precision == Precision.FP16;
    }
}