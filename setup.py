import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="piidetect",
    version="0.0.0.2",
    author="Edward Lu",
    author_email="maxminicherrycc@gmail.com",
    description="A package to build an end-to-end ML pipeline to detect personally identifiable information from text. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edwardcooper/piidetect",
    packages=setuptools.find_packages(),
    install_requires=[
          'gensim', 'numpy','pandas','faker','tqdm','sklearn'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
