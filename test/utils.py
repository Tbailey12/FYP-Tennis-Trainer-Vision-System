from pathlib import Path

def get_project_root():
    ''' Returns the path to the root project dir'''
    return Path(__file__).parent.parent