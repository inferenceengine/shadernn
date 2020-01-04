/* Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
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

import android.annotation.SuppressLint;
import android.graphics.ImageFormat;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.OutputConfiguration;

import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.CamcorderProfile;
import android.media.Image;
import android.media.ImageReader;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.Surface;

import androidx.annotation.NonNull;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class CameraPreview implements ImageReader.OnImageAvailableListener {
    private static final String TAG = "CameraPreview";
    private CameraDevice cameraDevice;
    private Handler backgroundHandler;
    private HandlerThread backgroundThread;
    private Size previewSize;
    private int rotationDegrees = 0;
    private Integer lowExposureCompensation = Integer.MAX_VALUE;
    private ImageReader imageReader;
    private ImageReader mMainYuvImageReader;
    private boolean firstPreview = true;
    /**
     * MediaRecorder
     */
    private MediaRecorder mMediaRecorder;

    private Size mVideoSize = new Size(1920, 1080);
    private String mNextVideoAbsolutePath;
    private boolean reno60fps = false;

    @SuppressLint("MissingPermission")
    void startCamera(CameraManager cameraManager, int displayRotation) {
        startBackgroundThread();
        try {
            String cameraId = cameraManager.getCameraIdList()[0];
            CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);
            // Find out if we need to swap dimension to get the preview size relative to sensor
            // coordinate.
            //noinspection ConstantConditions
            int sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            rotationDegrees = getRotationDegrees(displayRotation, sensorOrientation);
            Log.d(TAG, "displayRotation: " + displayRotation);
            Log.d(TAG, "sensorOrientation: " + sensorOrientation);
            Log.d(TAG, "rotationDegrees: " + rotationDegrees);
            int viewWidth = 1080;
            int viewHeight = 1920;
            boolean swappedDimensions = (rotationDegrees == 90 || rotationDegrees == 270);
            if (swappedDimensions) {
                int tmp = viewHeight;
                //noinspection SuspiciousNameCombination
                viewHeight = viewWidth;
                viewWidth = tmp;
            }
            StreamConfigurationMap map = characteristics.get(
                    CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (map != null) {
                Size[] outputSizes = map.getOutputSizes(ImageFormat.YUV_420_888);
                Size bestFit = new Size(0,0);
                double bestArea = getArea(bestFit);
                for (Size s : outputSizes) {
                    if (s.getWidth() <= viewWidth && s.getHeight() <= viewHeight) {
                        double area = getArea(s);
                        if (area > bestArea) {
                            bestFit = s;
                            bestArea = area;
                        }
                    }
                }
                previewSize = bestFit;
                Log.d(TAG, "Size: " + previewSize);
            }
            mNextVideoAbsolutePath = "/data/data/com.innopeaktech.seattle.snndemo/files/video.mp4";
            mMediaRecorder = new MediaRecorder();
            cameraManager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(@NonNull CameraDevice camera) {
                    cameraDevice = camera;
                    createCameraPreview();
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice camera) {

                }

                @Override
                public void onError(@NonNull CameraDevice camera, int error) {

                }
            }, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 0);
        ORIENTATIONS.append(Surface.ROTATION_90, 90);
        ORIENTATIONS.append(Surface.ROTATION_180, 180);
        ORIENTATIONS.append(Surface.ROTATION_270, 270);
    }

    private int getRotationDegrees(int displayRotationEnum, int sensorOrientationDegrees) {
        int displayRotationDegrees = ORIENTATIONS.get(displayRotationEnum);
        int diff = sensorOrientationDegrees - displayRotationDegrees;
        // round to nearest 90 degrees
        diff /= 45;
        if (diff % 2 == 1)
            diff += 1;
        diff *= 45;
        return (diff + 360) % 360;
    }

    private static double getArea(Size size) {
        return (double) size.getWidth() * (double) size.getHeight();
    }

    private void startBackgroundThread() {
        backgroundThread = new HandlerThread("Camera Preview");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    private void stopBackgroundThread() {
        if (backgroundThread != null) {
            backgroundThread.quitSafely();
            try {
                backgroundThread.join();
                backgroundThread = null;
                backgroundHandler = null;
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
    private void setUpMediaRecorder() throws IOException {
        mMediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        mMediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mMediaRecorder.setOutputFile(mNextVideoAbsolutePath);
        CamcorderProfile profile = CamcorderProfile.get(CamcorderProfile.QUALITY_1080P);
        mMediaRecorder.setVideoFrameRate(60);
        mMediaRecorder.setVideoSize(profile.videoFrameWidth, profile.videoFrameHeight);
        mMediaRecorder.setVideoEncodingBitRate(profile.videoBitRate);
        mMediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        mMediaRecorder.setCaptureRate(60);
        mMediaRecorder.setOrientationHint(ORIENTATIONS.get(0));
        mMediaRecorder.prepare();
    }

    ImageReader.OnImageAvailableListener mImageAvailListener = new ImageReader.OnImageAvailableListener() {
        @Override
        public void onImageAvailable(ImageReader reader) {
            // get image
            Image image = reader.acquireNextImage();

            //free the Image
            image.close();
        }
    };

    private void setUpSnapshotReader() {
        mMainYuvImageReader = ImageReader.newInstance(1920, 1080,
                ImageFormat.YUV_420_888, 2);
        mMainYuvImageReader.setOnImageAvailableListener(mImageAvailListener, backgroundHandler);
    }

    private void createCameraPreview() {
        try {
            /* Setup surfaces to get 1080p @ 60fps */
            if (firstPreview) {
                setUpMediaRecorder();
                firstPreview = false;
            }
            setUpSnapshotReader();

            final CaptureRequest.Builder requestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_VIDEO_SNAPSHOT);
            imageReader = ImageReader.newInstance(previewSize.getWidth(), previewSize.getHeight(), ImageFormat.YUV_420_888, 2);
            imageReader.setOnImageAvailableListener(this, backgroundHandler);

            Surface surface = imageReader.getSurface();
            Surface snapshotSurface = mMainYuvImageReader.getSurface();
            Surface recorderSurface = mMediaRecorder.getSurface();
            /* We only need one output */
            requestBuilder.addTarget(surface);
            //requestBuilder.addTarget(snapshotSurface);
            //requestBuilder.addTarget(recorderSurface);

//            RecordingManager rm = RecordingManager.getInstance();
//            Surface recordingSurface = rm.getSurface(previewSize.getWidth(), previewSize.getHeight());
//            requestBuilder.addTarget(recordingSurface);

            Log.w(TAG, "Model:" + Build.MODEL);
            String model = Build.MODEL;
            String reno10x = "PCCM00";
//            if(model.equals(reno10x)) {
//                Log.w(TAG, "Model is OPPO Phone (Reno 10X)... 60fps");
//                reno60fps = true;
//                List<OutputConfiguration> currentOutputs = new ArrayList<>();
//                currentOutputs.add(new OutputConfiguration(surface));
//                currentOutputs.add(new OutputConfiguration(snapshotSurface));
//                currentOutputs.add(new OutputConfiguration(recorderSurface));
//                cameraDevice.createCustomCaptureSession(null,
//                        currentOutputs,
//                        0x8021
//                    /*  0x8001 => OPERATING_MODE_CAPTURE
//                        0x800B => OPERATING_MODE_VIDEO --> Working
//                        0x800C => OPERATING_MODE_FAST_VIDEO
//                        0x8021 => OPERATING_MODE_VIDEO_HIGH_FPS : CapMode60FpsVideo */,
//                        new CameraCaptureSession.StateCallback() {
//                            @Override
//                            public void onConfigured(@NonNull CameraCaptureSession session) {
//                                if (cameraDevice == null)
//                                    return;
//
//                                requestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
//                                try {
//                                    Range<Integer> frameRate = Range.create(60, 60);
//                                    requestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, frameRate);
//                                    List<CaptureRequest> requestList = new ArrayList<>();
//                                    requestList.add(requestBuilder.build());
//
//                                    session.setRepeatingBurst(requestList, new CameraCaptureSession.CaptureCallback() {
//                                        @Override
//                                        public void onCaptureCompleted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull TotalCaptureResult result) {
//                                            super.onCaptureCompleted(session, request, result);
//                                            Long timeStamp = result.get(TotalCaptureResult.SENSOR_TIMESTAMP);
//                                            Integer exposureCompensation = result.get(TotalCaptureResult.CONTROL_AE_EXPOSURE_COMPENSATION);
//                                            if (exposureCompensation < lowExposureCompensation) {
//                                                lowExposureCompensation = exposureCompensation;
//                                            }
//                                            NativeLibrary.queueMetaData(timeStamp, exposureCompensation == lowExposureCompensation);
//                                        }
//                                    }, backgroundHandler);
//                                } catch (CameraAccessException e) {
//                                    e.printStackTrace();
//                                }
//                            }
//
//                            @Override
//                            public void onConfigureFailed(@NonNull CameraCaptureSession session) {
//
//                            }
//                        }, null);
//            } else {

                Log.w(TAG, "Model is not OPPO Phone... 30fps");
                cameraDevice.createCaptureSession(Arrays.asList(surface),
                        new CameraCaptureSession.StateCallback() {
                            @Override
                            public void onConfigured(@NonNull CameraCaptureSession session) {
                                if (cameraDevice == null)
                                    return;

                                requestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
                                try {
                                    session.setRepeatingRequest(requestBuilder.build(), new CameraCaptureSession.CaptureCallback() {
                                        @Override
                                        public void onCaptureCompleted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull TotalCaptureResult result) {
                                            super.onCaptureCompleted(session, request, result);
                                            Long timeStamp = result.get(TotalCaptureResult.SENSOR_TIMESTAMP);
                                            Integer exposureCompensation = result.get(TotalCaptureResult.CONTROL_AE_EXPOSURE_COMPENSATION);
                                            if (exposureCompensation < lowExposureCompensation) {
                                                lowExposureCompensation = exposureCompensation;
                                            }
                                            NativeLibrary.queueMetaData(timeStamp, exposureCompensation == lowExposureCompensation);
                                        }
                                    }, backgroundHandler);
                                } catch (CameraAccessException e) {
                                    e.printStackTrace();
                                }
                            }

                            @Override
                            public void onConfigureFailed(@NonNull CameraCaptureSession session) {

                            }
                        }, backgroundHandler);
    //        }
        } catch (CameraAccessException | IOException e) {
            e.printStackTrace();
        }
    }

    void stopCamera() {
        if (cameraDevice != null)
            cameraDevice.close();
        stopBackgroundThread();
    }

    @Override
    public void onImageAvailable(ImageReader reader) {
        Image image = reader.acquireNextImage();
        if (image != null) {
            //Log.i(TAG, "Image format " +  image.getFormat()  + "?" + ImageFormat.YUV_420_888);
            long timestamp = image.getTimestamp();
            Image.Plane[] planes = image.getPlanes();
            if ((planes.length != 3)) throw new AssertionError();
            int rowStride = planes[0].getRowStride();
            if ((rowStride != image.getWidth())) throw new AssertionError();

            Image.Plane yPlane = planes[0];
            if (yPlane.getPixelStride() != 1) throw new AssertionError();
            ByteBuffer yBuffer = yPlane.getBuffer();

            Image.Plane uPlane = planes[1];
            Image.Plane vPlane = planes[2];
            if (reno60fps) {
                // For some reason, the UV planes get swapped in this capture mode.
                // In the normal capture mode, both on this and other devices (i.e. Pixel 4), the
                // U and V planes are interlaced in VU order (NV21 format).  In NV21 format
                // Image.getPlanes()[1] should represent the U channel and start with the second
                // byte of the UV block, while Image.getPlanes()[2] should represent the V channel
                // and start with the first byte.
                // In this capture mode, however, the U and V values are stored in UV order
                // (NV12 format), which itself isn't so bad, except that planes returned
                // by Image.getPlanes() still point to the same bytes they would if the frame was
                // in NV21 format.  This means that planes[1] actually represents the V
                // channel, while planes[2] actually represents the U channel.
                // This is at odds with the Android documentation for ImageFormat.YUV_420_888:
                //  "The order of planes in the array returned by Image#getPlanes() is guaranteed
                //  such that plane #0 is always Y, plane #1 is always U (Cb), and plane #2 is
                //  always V (Cr)."
                //  https://developer.android.com/reference/android/graphics/ImageFormat#YUV_420_888
                uPlane = planes[2];
                vPlane = planes[1];
            }
            if ((uPlane.getRowStride() != rowStride)) throw new AssertionError();
            int pixelStride = uPlane.getPixelStride();
            if ((vPlane.getPixelStride() != pixelStride)) throw new AssertionError();
            if (pixelStride != 2) throw new AssertionError();
            ByteBuffer uBuffer = uPlane.getBuffer();
            ByteBuffer vBuffer = vPlane.getBuffer();

            NativeLibrary.queueFrame(image.getWidth(), image.getHeight(), rotationDegrees, timestamp, yBuffer, uBuffer, vBuffer);
            image.close();
        }
    }
}
