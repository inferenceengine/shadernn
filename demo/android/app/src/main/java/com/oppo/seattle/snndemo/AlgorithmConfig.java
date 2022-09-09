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

    public enum Pipeline {
        CLEARANCE1FRAME,
        CLEARANCE2FRAME,
        HYBRIDDL,
        BASIC_CNN
    }

    public enum DenoiserAlgorithm {
        NONE, AIDENOISER, SPATIALDENOISER
    }

    public enum Denoiser {
        NONE, COMPUTESHADER, FRAGMENTSHADER
    }

    public enum Clearance {
        NONE, SPRINT3CPU, SPRINT3DSP, SPRINT4CPU, SPRINT4DSP, SPRINT5CPU, SPRINT5DSP
    }

    public enum ClassifierAlgorithm {
        NONE, RESNET18, MOBILENETV2
    }

    public enum Classifier {
        COMPUTESHADER, FRAGMENTSHADER
    }

    public enum DetectionAlgorithm {
        NONE, YOLOV3
    }

    public enum Detection {
        COMPUTESHADER, FRAGMENTSHADER
    }

    public enum StyleTransfer {
        NONE, CANDY, MOSAIC, POINTILISM, RAIN_PRINCESS, UDNIE
    }

    public int classifierIndex;
    private String[] resnet18Classes = {"None", "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    private String[] mobilenetClasses = {"None", "Class 1", "Class 2"};

    private Pipeline pipeline;
    private Denoiser denoiser;
    private DenoiserAlgorithm denoiserAlgorithm;
    private Clearance clearance;
    private ClassifierAlgorithm classifierAlgorithm;
    private Classifier classifier;
    private DetectionAlgorithm detectionAlgorithm;
    private Detection detection;
    private boolean temporalFilter;
    private StyleTransfer styleTransferAlgorithm;

    AlgorithmConfig() {
        init();
    }

    private void init() {
        pipeline = Pipeline.CLEARANCE1FRAME;
        denoiserAlgorithm = DenoiserAlgorithm.NONE;
        denoiser = Denoiser.FRAGMENTSHADER;
        clearance = Clearance.NONE;
        classifierAlgorithm = ClassifierAlgorithm.NONE;
        classifier = Classifier.FRAGMENTSHADER;
        detectionAlgorithm = DetectionAlgorithm.NONE;
        detection = Detection.FRAGMENTSHADER;
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

    public void setPipeline(Pipeline pipeline) {
        this.pipeline = pipeline;
    }

    public Pipeline getPipeline() {
        return this.pipeline;
    }

    void setDenoiserAlgorithm(DenoiserAlgorithm denoiserAlgorithm) {
        this.denoiserAlgorithm = denoiserAlgorithm;
    }

    public Denoiser getDenoiser() {
        return this.denoiser;
    }

    public DenoiserAlgorithm getDenoiserAlgorithm() {
        return  this.denoiserAlgorithm;
    }

    void setStyleTransferAlgorithm(StyleTransfer styleTransferAlgorithm) {
        this.styleTransferAlgorithm = styleTransferAlgorithm;
    }

    void setSSBO(boolean ssbo) {
        //this.ssbo = ssbo;
    }

    public void setClearance(Clearance clearance) {
        this.clearance = clearance;
    }

    public Clearance getClearance() {
        return this.clearance;
    }

    void setTemporalFilter(boolean temporalFilter) {
        this.temporalFilter = temporalFilter;
    }

    public boolean getTemporalFilter() {
        return this.temporalFilter;
    }

    public boolean isDenoiseNONE() {
        return denoiserAlgorithm == DenoiserAlgorithm.NONE;
    }

    public boolean isDenoiseSPATIALDENOISER() {return  denoiserAlgorithm == DenoiserAlgorithm.SPATIALDENOISER; }

    public boolean isStyleTransferNONE() { return styleTransferAlgorithm == StyleTransfer.NONE; }

    public boolean isStyleTransferCANDY() { return styleTransferAlgorithm == StyleTransfer.CANDY; }

    public boolean isStyleTransferMOSAIC() { return styleTransferAlgorithm == StyleTransfer.MOSAIC; }

    public boolean isStyleTransferPOINTILISM() { return styleTransferAlgorithm == StyleTransfer.POINTILISM; }

    public boolean isStyleTransferRAINPRINCESS() { return styleTransferAlgorithm == StyleTransfer.RAIN_PRINCESS; }

    public boolean isStyleTransferUDNIE() { return  styleTransferAlgorithm == StyleTransfer.UDNIE; }

    public ClassifierAlgorithm getClassifierAlgorithm() {return classifierAlgorithm;}

    public void setClassifierAlgorithm(ClassifierAlgorithm classifierAlgorithm) {
        this.classifierAlgorithm = classifierAlgorithm;
        this.classifierIndex = 0;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
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

    public DetectionAlgorithm getDetectionAlgorithm() {
        return this.detectionAlgorithm;
    }

    public void setDetectionAlgorithm(DetectionAlgorithm detectionAlgorithm) {
        this.detectionAlgorithm = detectionAlgorithm;
    }

    public Detection getDetection() {
        return detection;
    }

    public void setDetection(Detection detection) {
        this.detection = detection;
    }

    public boolean isDetectionNONE() {
        return this.detectionAlgorithm == DetectionAlgorithm.NONE;
    }

    public boolean isDetectionYolov3() {
        return this.detectionAlgorithm == DetectionAlgorithm.YOLOV3;
    }
}