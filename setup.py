import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LattyMorph",
    version="1.1.0",
    author="Dominik Dold (ESA, Uni Vienna); Nicole Rosi (ESA)",
    author_email="",
    description="Package including methods for objective-driven morphing of Totimorphic lattices.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.8',
    install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'jupyter',
          'torch',
          'tqdm',
          'dgl==1.0.1',
          'sortedcontainers',
          'seaborn',
    ]
)
