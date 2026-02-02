#!/usr/bin/env python3

from setuptools import setup # type: ignore
import up_social_laws


long_description=\
'''
 ============================================================
    SOCIAL_LAWS
 ============================================================

    up_social_laws is a package that allows for various compilations and checks related to social law verification and synthesis.
'''

setup(name='up_social_laws',
      version=up_social_laws.__version__,
      description='Unified Planning Integration of Social Laws',
      long_description='Integration of Social Laws into the Unified Planning Framework',
      author='Technion Cognitive Robotics Lab',
      author_email='karpase@technion.ac.il',
      url='https://github.com/aiplan4eu/up-social-laws',
      classifiers=['Development Status :: 4 - Beta',
               'License :: OSI Approved :: Apache Software License',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering :: Artificial Intelligence'
               ],
      packages=['up_social_laws'],
      install_requires=['unified-planning', 'pandas'],
      python_requires='>=3.7',
      license='APACHE'
      )
