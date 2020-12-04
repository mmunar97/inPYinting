from setuptools import setup

setup(
    name='inPYinting',
    version='1.0',
    packages=['inPYinting'],
    url='https://github.com/mmunar97',
    license='mit',
    author='marcmunar',
    author_email='marc.munar@uib.es',
    description='Set of algorithms for image reconstruction and inpainting',
    include_package_data=True,
    install_requires=[
        "numpy",
        "opencv-python",
        "softcolor",
        "scikit-image"
    ]
)
