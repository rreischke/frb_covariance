from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()



setup(name='frb_cov',
      version='1.0',
      description='Covariance between localised fast radio bursts',
      url='',
      author='Robert Reischke',
      author_email='reischke@posteo.net',
      packages=['frb_cov'],
      zip_safe=False)
