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
import android.os.Bundle;
import android.util.Log;

public class CustomInstrumentationRunner extends androidx.test.runner.AndroidJUnitRunner {
    @Override
    public void onCreate(Bundle arguments) {
        super.onCreate(arguments);
        String numLoopsArg = (String) arguments.get("num_loops");
        Log.i("SNN", String.format("num_loops = %s", numLoopsArg));
        try {
            int numLoops = Integer.parseInt(numLoopsArg);
            NativeTests.setNumLoops(numLoops);
        } catch (NumberFormatException ex) {
            Log.e("SNN", "Incorrect value of num_loops parameter");
        }
    }
}