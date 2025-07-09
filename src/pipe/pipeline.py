import os
import sys


def main():
    # Get the absolute path to the current script's directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocessing = os.path.join(base_dir, 'preprocessing', 'join.py')
    modelling = os.path.join(base_dir, 'modelling', 'modelling.py')
    validate = os.path.join(base_dir, 'validate', 'validate.py')

    print('Running preprocessing...')
    os.system(f'python "{preprocessing}"')
    print('Running modelling...')
    os.system(f'python "{modelling}"')
    print('Running validation...')
    os.system(f'python "{validate}"')
    print('Pipeline complete.')

if __name__ == "__main__":
    main()
