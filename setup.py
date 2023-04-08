import setuptools
import matplotlib

setuptools.setup(
    name="pyhailing",
    version="0.0.9",
    url="https://github.com/nkullman/ridehailing_package",
    author="Nicholas Kullman",
    author_email="nick.kullman@gmail.com",
    description="An OpenAI gym environment for a ridehailing application",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "gym",
        "matplotlib",
        "numpy>=1.19,<1.22",
        "pandas",
        "Pillow",
        "scipy",
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: Apache Software License",
    ],
    include_package_data=True,
    package_data={'': ['data/*']},
    python_requires=">=3.7",
)
