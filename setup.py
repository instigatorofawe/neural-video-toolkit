from setuptools import setup, find_packages

setup(
    name='neural-video-toolkit',
    version='0.1.1',
    packages=find_packages(),
    url='https://github.com/instigatorofawe/neural-video-toolkit',
    license='MIT',
    author='Ran Liu',
    author_email='xabsox@gmail.com',
    description='A toolkit for integrating neural video processing methods into an FFMPEG pipeline.',
    scripts=['scripts/upscale.py']
)
