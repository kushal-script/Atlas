"""Installable package metadata, so the localizer can be imported from any
environment with `pip install -e .`; the entry points keep working from the
repository without installation because they add src to the path themselves."""

from setuptools import find_packages, setup

setup(
    name="drift_sense",
    version="0.1.0",
    description="Reference to search registration for SEM navigation error recovery",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=["numpy", "scipy", "opencv-python", "pillow"],
    include_package_data=False,
)
