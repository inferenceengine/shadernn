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

import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.ListView;

import androidx.preference.EditTextPreference;
import androidx.preference.Preference;
import androidx.preference.PreferenceFragmentCompat;
import androidx.preference.PreferenceManager;
import androidx.preference.SwitchPreference;
import androidx.recyclerview.widget.DividerItemDecoration;
import androidx.recyclerview.widget.RecyclerView;

import java.io.File;
import java.util.Map;

import static android.app.Activity.RESULT_OK;

public class MySettingsFragment extends PreferenceFragmentCompat implements SharedPreferences.OnSharedPreferenceChangeListener {
    private static final String TAG = "MySettingsFragment";
    SharedPreferences sharedPreferences;
    static final int PICK_FOLDER_PATH = 1;

    @Override
    public void onCreatePreferences(Bundle savedInstanceState, String rootKey) {
        setPreferencesFromResource(R.xml.preference_main, rootKey);

        //Bind on preference click listener
        initDefaultSavePathListener();


//        RecyclerView recyclerView = getListView();
//        DividerItemDecoration itemDecoration = new DividerItemDecoration(getContext(), RecyclerView.VERTICAL);
//        recyclerView.addItemDecoration(itemDecoration);

        //Default Save Path change listener
//        bindPreferenceSummaryToValue(findPreference(getString(R.string.default_save_path)));

        //Show summary value
//        showSummaryValue(findPreference(getString(R.string.default_save_path)));
    }

    @Override
    public void onResume() {
        super.onResume();
        sharedPreferences = getPreferenceManager().getSharedPreferences();
        sharedPreferences.registerOnSharedPreferenceChangeListener(this);
        //Show default preferences as summary
        Map<String, ?> preferencesMap = sharedPreferences.getAll();
        for (Map.Entry<String, ?> preferenceEntry : preferencesMap.entrySet()) {
            if (preferenceEntry instanceof EditTextPreference) {
                updateSummary((EditTextPreference) preferenceEntry);
            }
        }
        EditTextPreference editTextPreference = findPreference(getString(R.string.key_default_save_path));
        editTextPreference.setSummary(sharedPreferences.getString(getString(R.string.key_default_save_path), ""));
    }

    @Override
    public void onPause() {
        sharedPreferences.unregisterOnSharedPreferenceChangeListener(this);
        super.onPause();
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        Log.d(TAG, "onSharedPreferenceChanged");
        Map<String, ?> preferencesMap = sharedPreferences.getAll();
        Object changedPreference = preferencesMap.get(key);
        if (preferencesMap.get(key) instanceof EditTextPreference) {
            updateSummary((EditTextPreference) changedPreference);
        }
    }

    private void updateSummary(EditTextPreference preference) {
        preference.setSummary(sharedPreferences.getString(getString(R.string.key_default_save_path), ""));
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == PICK_FOLDER_PATH) {
            if (resultCode == RESULT_OK) {
                Uri uri = data.getData();
                File file = new File(uri.getPath());
                String newPathName = file.getAbsolutePath();
                String newFolderName = file.getName();
                EditTextPreference preference = findPreference(getString(R.string.key_default_save_path));
                preference.setText(newPathName);
                preference.setSummary(newPathName);
            }
        }
    }

    @Override
    public void onDisplayPreferenceDialog(Preference preference) {
        if (preference.getKey().equals(getString(R.string.key_default_save_path))){

        } else
            super.onDisplayPreferenceDialog(preference);
    }

    private void initDefaultSavePathListener () {
        Preference defaultSavePath = findPreference(getString(R.string.key_default_save_path));
        defaultSavePath.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener(){

            @Override
            public boolean onPreferenceClick(Preference preference) {
                Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
                startActivityForResult(intent, PICK_FOLDER_PATH);
                return true;
            }
        });
    }

    private void bindPreferenceSummaryToValue(Preference preference) {
        preference.setOnPreferenceChangeListener(sBindPreferenceSummaryToValue);
        sBindPreferenceSummaryToValue.onPreferenceChange(preference,
                PreferenceManager
        .getDefaultSharedPreferences(preference.getContext())
        .getString(preference.getKey(), ""));
    }

    private Preference.OnPreferenceChangeListener sBindPreferenceSummaryToValue = new Preference.OnPreferenceChangeListener() {
        @Override
        public boolean onPreferenceChange(Preference preference, Object newValue) {
            String stringValue = newValue.toString();

            if (preference instanceof EditTextPreference) {
                if (preference.getKey().equals("default_save_path")) {
                    preference.setSummary(stringValue);
                }
            } else {
                preference.setSummary(stringValue);
            }
            return true;
        }
    };
}
