from os import mkdir, path, getcwd,listdir

def get_subdirectory(sd):
    dir = path.join(getcwd(), sd)
    if not path.isdir(dir):
        mkdir(dir)
    return dir

def get_all_elements_in_dir(sd, filter=lambda x: x):
    return  [file for file in listdir(sd) if filter(file)]
