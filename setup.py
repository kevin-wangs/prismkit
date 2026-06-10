from setuptools import setup, find_packages

setup(
    name="prismkit",
    version="0.1.0",
    description="Example projects & tutorials for the Optrix compute ecosystem",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kevin Wangs",
    url="https://github.com/kevin-wangs/prismkit",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "optrix>=0.1.0",
        "celatrix>=0.1.0",
        "novastm>=0.1.0",
        "spectune>=0.1.0",
        "voxclad>=0.1.0",
        "numpy>=1.24",
    ],
)
