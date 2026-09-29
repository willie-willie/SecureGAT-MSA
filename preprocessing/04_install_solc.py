#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:29:23 2026

@author: Willie
"""


from solcx import install_solc, set_solc_version, get_installed_solc_versions


print("="*60)
print("Solidity Compiler Setup")
print("="*60)


version = "0.5.17"


print("\nInstalling Solidity compiler:")
print(version)


install_solc(version)


set_solc_version(version)


print("\nInstalled compilers:")

for compiler in get_installed_solc_versions():

    print(compiler)


print("\nSolc setup completed")