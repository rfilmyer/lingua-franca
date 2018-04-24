import os
import shutil

languages = next(os.walk("zips"))[1]
for language in languages:
    if not os.path.exists(language):
        os.makedirs(language)
    uploads = next(os.walk(os.path.join("zips", language)))[1]
    for upload_name in uploads:
        full_folder_name = os.path.join("zips", language, upload_name)
        if os.path.exists(os.path.join(full_folder_name, "wav")):
            for wavfile in os.listdir(os.path.join(full_folder_name, "wav")):
                orig_path = os.path.join(full_folder_name, "wav", wavfile)
                shutil.copyfile(orig_path, os.path.join(language, upload_name + "_" + wavfile))