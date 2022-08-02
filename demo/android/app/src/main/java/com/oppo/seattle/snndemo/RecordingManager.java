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

import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.Resources;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaFormat;
import android.media.MediaRecorder;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Surface;

import androidx.preference.PreferenceManager;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Allows us to record video.
 */
public class RecordingManager implements
        SharedPreferences.OnSharedPreferenceChangeListener,
        CustomDialogBuilder.CustomDialogListener {

    private static final String TAG = "RecordingManager";

    private static final String MIME_TYPE = "video/mp4";
    private static final int BIT_RATE = 8000000;
    private static final int FRAME_RATE = 60;
    private static final int IFRAME_INTERVAL = 3;
    private static final int WIDTH = 1080;
    private static final int HEIGHT = 1920;

    private static RecordingManager instance;
    private Context context;
    private boolean isRecording;
    private boolean saveAs;
    private String path;
    private MediaCodec codec;

    /**
     * Recorder being used to generate video.
     */
    private MediaRecorder recorder;

    /**
     * The surface of the video recorder.
     */
    private Surface surface;

    /**
     *
     * @param context
     */
    private RecordingManager(Context context) {
        this.context = context;
        isRecording = false;
        initPreferences();
    }

    /**
     * If this singleton has not already been initialized,
     * this method will create it.
     * @return The instance of this singleton.
     */
    public static RecordingManager getInstance(Context context) {
        if (instance == null) {
            synchronized (RecordingManager.class) {
                if (instance == null) {
                    instance = new RecordingManager(context);
                }
            }
        }
        return instance;
    }

    /**
     * This method will NOT initialize the singleton if it hasn't been already.
     * @return The instance of this singleton.
     */
    public static RecordingManager getInstance() {
        assert instance != null;
        return instance;
    }

    /**
     * Starts recording the processed images to a new video.
     * Method call is ignored if it is already recording.
     */
    void startRecording() {
        //If this is not already recording.
        if (!isRecording) {
            //Record that we have started recording.
            isRecording = true;

            //Create the video recorder.
            initRecorder(WIDTH, HEIGHT);

            //Start recording the images to the video.
            NativeLibrary.startRecording(surface);
            recorder.start();
            Log.d(TAG, "Started Recording");
        }
    }

    /**
     * Stops recording video.
     * Method call is ignored if it isn't recording.
     */
    void stopRecording() {
        //If this isn't already stopped.
        if (isRecording) {
            //Record that we are no longer recording.
            isRecording = false;
            NativeLibrary.stopRecording();
            try {
                recorder.stop();
                Log.d(TAG, "Stopped Recording");
            } catch (RuntimeException r) {
                Log.e(TAG, "Either video/audio was not detected");
                r.printStackTrace();
                // TODO: delete the file created for recording to.
            }
            recorder.release();
            if (saveAs) {
                promptSaveAs();
            } else {
                addRecordingToMediaLibrary("coolFileName");
            }
            recorder = null;
        }
    }

    /**
     * Initializes the object that will create the video file.
     * @param width
     * @param height
     */
    private void initRecorder(int width, int height) {
        try {
            if (recorder == null) {
                recorder = new MediaRecorder();
            }

            //Decide where to save the video.
            File file = new File(context.getExternalFilesDir(null), "video.mp4");
            Log.d(TAG, file.getAbsolutePath());
            boolean createResult = file.createNewFile();

            //Setup the video recorder.
            recorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            recorder.setCaptureRate(45);
            recorder.setVideoFrameRate(FRAME_RATE);
            recorder.setVideoSize(width, height);
            recorder.setVideoEncodingBitRate(BIT_RATE);
            recorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
            recorder.setOutputFile(file);
            recorder.prepare();

            surface = recorder.getSurface();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void setSurface(Surface surface, int width, int height) {
        initRecorder(width, height);
        recorder.setInputSurface(surface);
    }

    public Surface getSurface(int width, int height) {
        initRecorder(width, height);
        return surface;
    }

    public Surface getEncoderSurface() {
        MediaFormat format = createMediaFormat();
        MediaCodecInfo mediaCodecInfo = selectCodecType(MIME_TYPE);
        try {
            assert mediaCodecInfo != null;
            codec = MediaCodec.createByCodecName(mediaCodecInfo.getName());
        } catch (IOException e) {
            e.printStackTrace();
        }
        Surface inputSurface = MediaCodec.createPersistentInputSurface();
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        codec.setInputSurface(inputSurface);
        return inputSurface;
    }

    private static MediaFormat createMediaFormat() {
        MediaFormat format = MediaFormat.createVideoFormat(MIME_TYPE, WIDTH, HEIGHT);
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
        format.setInteger(MediaFormat.KEY_BIT_RATE, BIT_RATE);
        format.setInteger(MediaFormat.KEY_FRAME_RATE, FRAME_RATE);
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, IFRAME_INTERVAL);
        return format;
    }

    private MediaCodecInfo selectCodecType(String type) {
        MediaCodecList list = new MediaCodecList(MediaCodecList.ALL_CODECS);
        MediaCodecInfo[] codecInfos = list.getCodecInfos();
        for (MediaCodecInfo codecInfo : codecInfos) {
            if (!codecInfo.isEncoder()) {
                continue;
            }
            String[] types = codecInfo.getSupportedTypes();
            for (String s : types) {
                if (s.equalsIgnoreCase(type)) {
                    return codecInfo;
                }
            }
        }
        return null;
    }

    private void promptSaveAs() {
        CustomDialogBuilder cdb = new CustomDialogBuilder(context);
        cdb.requestSaveAs("Save As", null);
    }

    private void addRecordingToMediaLibrary(String filename) {
        Log.e(TAG, "Entering addRecordingToMediaLibrary");
        if (filename.equals("")) filename = getDefaultNamePattern();
        ContentValues values = new ContentValues(4);
        values.put(MediaStore.Video.Media.TITLE, filename);
        values.put(MediaStore.Video.Media.DISPLAY_NAME, filename);
        values.put(MediaStore.Video.Media.MIME_TYPE, MIME_TYPE);
        Uri uri = context.getContentResolver().insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values);
        File f = new File(context.getExternalFilesDir(null), "video.mp4");
        try {
            InputStream is = new FileInputStream(f);
            assert uri != null;
            OutputStream os = context.getContentResolver().openOutputStream(uri);
            byte[] buffer = new byte[4096];
            int len;
            while ((len = is.read(buffer)) != -1) {
                assert os != null;
                os.write(buffer, 0, len);
            }
            assert os != null;
            os.flush();
            is.close();
            os.close();
        } catch (Exception e) {
            Log.e(TAG, "exception while writing video");
        }
        context.sendBroadcast(new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, uri));
    }

    private String getDefaultNamePattern() {
        return String.valueOf(System.currentTimeMillis());
    }

    @Override
    public void onCustomDialogReturnValue(String result) {
        addRecordingToMediaLibrary("somethingCool");
    }

    private void updateSavePath(String path) {
//        NativeLibrary.updateSavePath(path);
    }

    private void initPreferences() {
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
        sharedPreferences.registerOnSharedPreferenceChangeListener(this);
        Resources resources = context.getResources();
        saveAs = sharedPreferences.getBoolean(resources.getString(R.string.key_toggle_save_as), false);
        Log.d(TAG, "Preference Save As: " + saveAs);
        path = sharedPreferences.getString(resources.getString(R.string.key_default_save_path), "");
        Log.d(TAG, "Preference Save Path: " + path);
        updateSavePath(path);
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        Log.d(TAG, "onSharedPreferenceChanged() called");
        Resources resources = context.getResources();
        if (key.equals(resources.getString(R.string.key_default_save_path))) {
            String newPath = sharedPreferences.getString(key, "");
            Log.d(TAG, "Preference Save Path: " + newPath);
            updateSavePath(newPath);
        } else if (key.equals(resources.getString(R.string.key_toggle_save_as))) {
            saveAs = sharedPreferences.getBoolean(key, false);
            Log.d(TAG, "Preference Save As: " + saveAs);
        }
    }

    /**
     *
     * @return true if this is currently recording, false otherwise.
     */
    public boolean isRecording() {
        return this.isRecording;
    }

}
