from setuptools import setup, find_packages

setup(
    name='pauc',
    version='0.1.1',
    packages=find_packages(),
    description='Compute ROC AUC and confidence intervals using DeLong’s method',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'scipy',
        'pandas'
    ],
    tests_require=[
        'pytest',
    ],
    python_requires='>=3.6',
    author='Srijit Seal',
    author_email='srijit@understanding.bio',
    url='https://github.com/srijitseal/pauc',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
