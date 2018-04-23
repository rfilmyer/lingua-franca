import os

from typing import List, Tuple, Union
FilePath = Union[str, bytes]


def get_files(voxforge_directory: FilePath = "voxforge") -> List[Tuple[FilePath, str]]:
    """
    Discovers audio files in the voxforge directory and returns their paths and languages.

    :return: A list of tuples containing the file path of a soundfile and its language.
    """
    file_list = []

    language_folders = next(os.walk(voxforge_directory))[1]
    for language_folder in language_folders:
        language_folder_path = os.path.join(voxforge_directory, language_folder)

        files_in_language_folder = next(os.walk(language_folder_path))[2]
        for soundfile in files_in_language_folder:
            soundfile_path = os.path.join(language_folder_path, soundfile)

            file_list.append((soundfile_path, language_folder))

    return file_list

if __name__ == "__main__":

    for entry in get_files():
        print(entry)


