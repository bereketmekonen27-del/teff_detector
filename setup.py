from setuptools import setup, find_packages

setup(
    name="teff_detector",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "teff_detector": ["teff_disease_model.pth"],
    },
    install_requires=[
        "torch",
        "torchvision",
        "Pillow",
    ],
)
