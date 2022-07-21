#!/usr/bin/env python
#import os
#import sys
#from setuptools import setup

#setup(name='pds-ml',
#      packages=['src/self_supervised_learner', 
#      'src/utilities'],
#      )


from setuptools import setup
setup(name='pds-ml',
version='0.0.2',
description='A set of tools for PDS ML analysis.',
url='https://github.com/jcsmithhere/pds-ml',
author=['Jeff Smith'],
author_email=['jeffsmithnasapipelines@gmail.com'],
packages=['pds/self_supervised_learner', 'pds/utilities'],
#packages=['src'],
zip_safe=False)
