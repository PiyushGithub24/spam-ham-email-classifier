from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirement
    '''
    Hypen_E_Dot='-e .'
    requirements=[]
    with open (file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]

    if (Hypen_E_Dot in requirements):
        requirements.remove(Hypen_E_Dot)
    return requirements


setup(
    name="Spam-Ham-Email-Classifier",
    version="0.0.1",
    author="Piyush",
    author_email="piyushrana3612@gmail.com",
    packages=find_packages(),
    install_requires=['numpy','pandas','seaborn']

)