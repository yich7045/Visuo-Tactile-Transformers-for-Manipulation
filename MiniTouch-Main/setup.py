import setuptools

setuptools.setup(
    name="minitouch",
    version="0.0.1",
    description="MiniTouch benchmarl",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=['gym', 'pybullet'],
    python_requires=">=3.6",
)