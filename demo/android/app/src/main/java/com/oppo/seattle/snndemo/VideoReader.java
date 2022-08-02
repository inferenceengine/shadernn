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
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.util.Log;

import androidx.annotation.NonNull;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;

class VideoReader {
    public static final int MAX_WIDTH = 1080;
    public static final int MAX_HEIGHT = 1920;
    public static final String TAG = "VideoReader";
    private MediaExtractor extractor;
    private MediaCodec decoder;
    private int rotation = 0;
    private int videoWidth = MAX_WIDTH;
    private int videoHeight = MAX_HEIGHT;
    private MainActivity mainActivity;

    /**
     * The first frame this read.
     * Is used to determine which frame
     * is low exposure.
     */
    private ByteBuffer frame0;

    private long frame0Timestamp;

    public VideoReader(MainActivity mainActivity) {
        this.mainActivity = mainActivity;
    }

    synchronized void startVideo(Uri newVideoUri) {
        closeVideo();
        try {
            openVideo(newVideoUri);
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        decoder.setCallback(new MediaCodec.Callback() {
            private int frameCount = 0;
            private double firstFrameSum = 0.0;
            private long firstFrameTime = 0;
            private boolean lowExposure = false;

            @Override
            public void onInputBufferAvailable(@NonNull MediaCodec codec, int index) {
                Log.d(TAG, "inputBuffer index: " + index);
                if (index >= 0) {
                    ByteBuffer inputBuffer = codec.getInputBuffer(index);
                    if (inputBuffer != null) {
                        synchronized (VideoReader.this) {
                            int sampleSize = extractor.readSampleData(inputBuffer, 0);
                            Log.d(TAG, "sampleSize: " + sampleSize);
                            if (sampleSize >= 0) {
                                codec.queueInputBuffer(index, 0, sampleSize, extractor.getSampleTime(), 0);
                                extractor.advance();
                            } else {
                                Log.w(TAG, "sampleSize < 0");
                                mainActivity.videoStopped();
                                closeVideo();
                            }
                        }
                    } else {
                        Log.w(TAG, "inputBufferr == null");
                        closeVideo();
                    }
                } else {
                    Log.w(TAG, "index < 0");
                    closeVideo();
                }
            }

            @Override
            public void onOutputBufferAvailable(@NonNull MediaCodec codec, int index, @NonNull MediaCodec.BufferInfo info) {
                Log.d(TAG, "outputBuffer index: " + index);
                ByteBuffer outputBuffer = codec.getOutputBuffer(index);
                MediaFormat bufferFormat = codec.getOutputFormat(index);
                int width = bufferFormat.getInteger(MediaFormat.KEY_WIDTH);
                int height = bufferFormat.getInteger(MediaFormat.KEY_HEIGHT);
                logMediaFormat(bufferFormat, "Frame Format: ");

                switch(frameCount) {
                    case 0:
                        //Create a copy of frame 0 that will survive the next call.
                        //Output buffer must not be null.
                        Objects.requireNonNull(outputBuffer);

                        //Create a deep copy of output buffer that will survive to the next call.
                        ByteBuffer outputBufferCopy = ByteBuffer.allocateDirect(
                                outputBuffer.remaining()
                        );
                        outputBufferCopy.put(outputBuffer);

                        //Cache frame 0 so we can compare it to the next frame.
                        frame0 = outputBufferCopy;
                        frame0Timestamp = info.presentationTimeUs;
                        break;

                    case 1:
                        // now we have 2 frames. it is time to determine if frame 0 is low exposure.
                        lowExposure = NativeLibrary.compareFrameExposure(width, height, frame0, outputBuffer) <= 0;
                        queueFrame(width, height, frame0Timestamp, frame0);
                        // fallthrough intentionally. no break here.

                    default:
                        lowExposure = !lowExposure;
                        queueFrame(width, height, info.presentationTimeUs, outputBuffer);
                }

                frameCount++;
                codec.releaseOutputBuffer(index, true);
            }

            private void queueFrame(int width, int height, long timestamp, ByteBuffer byteBuffer) {
                // The following offsets are only correct for NV12 byte layout
                ByteBuffer yPlane = byteBuffer.duplicate();
                int uvOffset = width * height;
                byteBuffer.position(uvOffset);
                ByteBuffer uPlane = byteBuffer.slice();
                byteBuffer.position(uvOffset + 1);
                ByteBuffer vPlane = byteBuffer.slice();
                if (height > videoHeight) {
                    // Discard extra scanlines (i.e. 1080p videos can produce 1088-height frames)
                    height = videoHeight;
                }
                NativeLibrary.queueFrame(width, height, rotation, timestamp, yPlane, uPlane, vPlane);
                NativeLibrary.queueMetaData(timestamp, lowExposure);
            }

            @Override
            public void onError(@NonNull MediaCodec codec, @NonNull MediaCodec.CodecException e) {
                e.printStackTrace();
                synchronized (VideoReader.this) {
                    closeVideo();
                }
            }

            @Override
            public void onOutputFormatChanged(@NonNull MediaCodec codec, @NonNull MediaFormat format) {
                logMediaFormat(format, "Format Changed: ");
            }
        });
        decoder.start();
    }

    private void logMediaFormat(@NonNull MediaFormat format, String msg) {
        int width = format.getInteger(MediaFormat.KEY_WIDTH);
        int height = format.getInteger(MediaFormat.KEY_HEIGHT);
        int colorFormat = format.getInteger(MediaFormat.KEY_COLOR_FORMAT);
        Log.i(TAG, msg + width + " " + height + " " + colorFormat);
    }

    synchronized void stopVideo() {
        closeVideo();
    }

    private synchronized void openVideo(Uri uri) throws IOException {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        Context context = mainActivity.getApplicationContext();
        retriever.setDataSource(context, uri);
        rotation = Integer.valueOf(retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION));
        videoWidth = Integer.valueOf(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH));
        videoHeight = Integer.valueOf(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT));
        Log.d(TAG, "Video dimensions: " + videoWidth + "x" + videoHeight);
        extractor = new MediaExtractor();
        extractor.setDataSource(context, uri, null);

        int index = extractor.getTrackCount();
        Log.d("MediaCodecTag", "Track count: " + index);

        for (int i = 0; i < extractor.getTrackCount(); i++) {
            MediaFormat format = extractor.getTrackFormat(i);
            String mime = format.getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("video/")) {
                extractor.selectTrack(i);
                decoder = MediaCodec.createDecoderByType(mime);
                decoder.configure(format, null, null, 0);
                logMediaFormat(decoder.getOutputFormat(), "Output format: ");
                break;
            }
        }

        if (decoder == null) {
            Log.e("DecodeActivity", "Can't find video info!");
        }

    }

    private synchronized void closeVideo() {
        if (decoder != null) {
            decoder.stop();
            decoder.release();
            decoder = null;
        }
        if (extractor != null) {
            extractor.release();
            extractor = null;
        }
    }
}
