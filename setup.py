from setuptools import setup, find_packages
from pathlib import Path

def read_requirements():
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file) as f:
            return f.readlines()
    return []

setup(
    name="cup",
    version="0.0.1",
    description="A Python CLI/package for managing plotting routines with CAFAna-created trees",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mattia Sotgia",
    author_email="mattia.sotgia@ge.infn.it",
    url="https://github.com/mattiasotgia/TwoDValidations/tree/main/plotStuff",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "tuna = cup.cli:main", 
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=read_requirements(),
    include_package_data=True,
    zip_safe=False,
)
