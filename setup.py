from setuptools import setup

setup(
    name='inPYinting',
    version='1.1',
    packages=['inPYinting'],
    url='https://github.com/mmunar97/inPYinting',
    license='mit',
    author='marcmunar',
    author_email='marc.munar@uib.es',
    description='Set of algorithms for image reconstruction and inpainting',
    include_package_data=True,
    install_requires=[
        "scipy",
        "numpy",
        "opencv-python",
        "softcolor",
        "scikit-image",
        "pyFFTW"
    ]
)
