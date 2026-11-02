from setuptools import find_packages, setup

setup(
    name="eindex-callum",
    version="0.1.3",
    packages=find_packages(exclude=["tests", "tests.*", "bench"]),
    install_requires=["torch", "numpy"],
    extras_require={"test": ["pytest", "einops"]},  # einops: only for the vendored original used as the test oracle
    author="Callum McDougall",
    author_email="cal.s.mcdougall@gmail.com",
    description="einops-style tensor indexing (compile-once, gather-speed fork maintained by ARENA)",
    url="https://github.com/ARENA-education/eindex",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
