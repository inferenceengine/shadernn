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

    public enum FrameCount {
        ONE,
        TWO
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

    public enum BasicCNN {
        NONE, BASIC_CNN
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
    private FrameCount frameCount;
    private Denoiser denoiser;
    private DenoiserAlgorithm denoiserAlgorithm;
    private boolean ssbo;
    private Clearance clearance;
    private ClassifierAlgorithm classifierAlgorithm;
    private Classifier classifier;
    private DetectionAlgorithm detectionAlgorithm;
    private Detection detection;
    private boolean temporalFilter;
    private BasicCNN alexNet;
    private StyleTransfer styleTransferAlgorithm;

    AlgorithmConfig() {
        init();
    }

    private void init() {
        pipeline = Pipeline.CLEARANCE1FRAME;
        frameCount = FrameCount.ONE;
        denoiserAlgorithm = DenoiserAlgorithm.NONE;
        denoiser = Denoiser.FRAGMENTSHADER;
        clearance = Clearance.NONE;
        alexNet = BasicCNN.NONE;
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

    void setDenoiser(Denoiser denoiser) {
        this.denoiser = denoiser;
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
        this.ssbo = ssbo;
    }

    public boolean getSSBO() {
        return this.ssbo;
    }

    public void setClearance(Clearance clearance) {
        this.clearance = clearance;
    }

    public void setBasicCNN(BasicCNN model) { this.alexNet = model;}
    public BasicCNN getBasicCNN() {return this.alexNet; }

    public Clearance getClearance() {
        return this.clearance;
    }

    void setTemporalFilter(boolean temporalFilter) {
        this.temporalFilter = temporalFilter;
    }

    public boolean getTemporalFilter() {
        return this.temporalFilter;
    }

    public boolean isFrameCountONE() {
        return this.frameCount == FrameCount.ONE;
    }

    public boolean isDenoiseNONE() {
        return denoiserAlgorithm == DenoiserAlgorithm.NONE;
    }

    public boolean isDenoiseFRAGMENTSHADER() {
        return denoiser == Denoiser.FRAGMENTSHADER;
    }

    public boolean isDenoiseCOMPUTESHADER() {
        return denoiser == Denoiser.COMPUTESHADER;
    }

    public boolean isDenoiseAIDENOISER() {return  denoiserAlgorithm == DenoiserAlgorithm.AIDENOISER; }

    public boolean isDenoiseSPATIALDENOISER() {return  denoiserAlgorithm == DenoiserAlgorithm.SPATIALDENOISER; }

    public boolean isSSBO() {
        return ssbo;
    }

    public boolean isClearanceNONE() {
        return clearance == Clearance.NONE;
    }

    public boolean isClearanceSPRINT3CPU() {
        return clearance == Clearance.SPRINT3CPU;
    }

    public boolean isClearanceSPRINT3DSP() {
        return clearance == Clearance.SPRINT3DSP;
    }

    public boolean isClearanceSPRINT4CPU() {
        return clearance == Clearance.SPRINT4CPU;
    }

    public boolean isClearanceSPRINT4DSP() {
        return clearance == Clearance.SPRINT4DSP;
    }

    public boolean isClearanceSPRINT5CPU() {
        return clearance == Clearance.SPRINT5CPU;
    }

    public boolean isClearanceSPRINT5DSP() {
        return clearance == Clearance.SPRINT5DSP;
    }

    public boolean isTemporalFilter() {
        return temporalFilter;
    }

    public boolean isBasicCNN() { return alexNet == BasicCNN.BASIC_CNN;}

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

    public boolean isClassifierFRAGMENTSHADER() {
        return this.classifier == Classifier.FRAGMENTSHADER;
    }

    public boolean isClassifierCOMPUTESHADER() {
        return this.classifier == Classifier.COMPUTESHADER;
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

    public boolean isDetectionFRAGMENTSHADER() {
        return this.detection == Detection.FRAGMENTSHADER;
    }

    public boolean isDetectionCOMPUTESHADER() {
        return this.detection == Detection.COMPUTESHADER;
    }
}