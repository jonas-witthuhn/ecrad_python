from setuptools import setup
setup(
    name="ecrad_python",
    version="1.1.1",
    description="Python scripts to convert CAMS input data to ecrad input and run ecrad simulations. Also functions to perturbe input for sensitivity studies.",
    url="https://github.com/jonas-witthuhn/ecrad_python-base",
    license="CC BY-NC",
    author="Jonas Witthuhn",
    author_email="witthuhn@tropos.de",
    packages=["ecrad_python"],
    package_dir={"":"src"},
    install_requires=["numpy",
                      "xarray",
                      "netcdf4",
                      "scipy",
                      "pandas",
                      "trosat-base @ git+https://github.com/hdeneke/trosat-base.git#egg=trosat-base"],
        )
