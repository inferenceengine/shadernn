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

import android.app.Dialog;
import android.content.Context;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.core.content.ContextCompat;

public class CustomDialogBuilder extends AlertDialog.Builder {
    private static final String TAG = "CDB";
    Context context;

    public interface CustomDialogListener {
        void onCustomDialogReturnValue(String result);
    }

    public CustomDialogBuilder(@NonNull Context context) {
        super(context);
        this.context = context;
    }

    public void requestSaveAs(String title, String defaultText) {
        AlertDialog.Builder builder = new AlertDialog.Builder(getContext());
        builder.setCustomTitle(getCustomTitle(title));
        EditText editTextHandle = getEditText(defaultText);
        builder.setView(getCustomEditText(editTextHandle));
        builder.setPositiveButton("OK", (dialog, which) -> {
            CustomDialogListener activity = (CustomDialogListener) context;
            activity.onCustomDialogReturnValue(editTextHandle.getText().toString());
        });
        builder.setNegativeButton("Cancel", (dialog, which) -> dialog.cancel());
        Dialog dialog = builder.create();
        dialog.show();
        Log.d(TAG, "Called dialog.show()");
    }



    private EditText getEditText(String defaultText) {
        EditText editText = new EditText(getContext());
        if (defaultText == null) defaultText = String.valueOf(System.currentTimeMillis());
        editText.setText(defaultText);
        editText.setLayoutParams(getLayoutParams(10, 10, 0, 0));
        editText.setOnTouchListener( (v, e) -> {editText.setText(""); return false;});
        return editText;
    }

    private View getCustomEditText(EditText editText) {
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
        params.leftMargin = 20; params.rightMargin = 20;
        FrameLayout backgroundFrame = new FrameLayout(getContext());
        backgroundFrame.setLayoutParams(getLayoutParams(20, 20, 50, 0));
        backgroundFrame.setBackgroundColor(ContextCompat.getColor(getContext(), R.color.editTextBackground));
        backgroundFrame.addView(editText);

        FrameLayout frameLayout = new FrameLayout(getContext());
        frameLayout.addView(backgroundFrame);
        return frameLayout;
    }

    private TextView getCustomTitle(String title) {
        TextView customTitle = new TextView(getContext());
        customTitle.setText(title);
        customTitle.setGravity(Gravity.CENTER_HORIZONTAL);
        customTitle.setTextSize(20f);
        return customTitle;
    }

    private ViewGroup.LayoutParams getLayoutParams(int leftMargin, int rightMargin, int topMargin, int bottomMargin) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
        params.leftMargin = leftMargin;
        params.rightMargin = rightMargin;
        params.topMargin = topMargin;
        params.bottomMargin = bottomMargin;
        return params;
    }
}
