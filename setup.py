from setuptools import setup, find_packages

setup(
    name="expfam",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "jaxopt",
        "jaxutil @ git+https://github.com/jonny-so/jaxutil.git#egg=jaxutil",
    ],
)