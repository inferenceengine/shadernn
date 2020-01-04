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

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;

import android.content.res.AssetManager;
import android.util.Log;

class DSP {
    private MainActivity mainActivity;
    static String TAG = "DSP";

    public DSP(MainActivity mainActivity) {
        this.mainActivity = mainActivity;
        try {
            AssetManager assetManager = mainActivity.getAssets();
            for(String dspLib : Arrays.asList(assetManager.list("dsp"))) {
                InputStream input = assetManager.open("dsp" + File.separator + dspLib);
                try {
                    File file = new File(mainActivity.getCacheDir(), dspLib);
                    try (OutputStream output = new FileOutputStream(file)) {
                        byte[] buffer = new byte[4 * 1024]; // or other buffer size
                        int read;

                        while ((read = input.read(buffer)) != -1) {
                            output.write(buffer, 0, read);
                        }

                        output.flush();
                    } catch (Exception e) {
                        Log.e(TAG, e.getMessage());
                    }
                } finally {
                    input.close();
                }
            }

            String skelLocation = mainActivity.getCacheDir().getPath();

            Log.i(TAG, "Skel library location : " + skelLocation);

            //Set the ADSP_LIBRARY_PATH to skelLocation
            NativeLibrary.initDSP(skelLocation);
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
        }
    }
}
