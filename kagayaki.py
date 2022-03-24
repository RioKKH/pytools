#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import toml
import traceback
from typing import Dict

import pexpect

def load(fin:str) -> Dict:
    with open(fin) as fr:
        dobj = toml.load(fr)
    return dobj

def main():
    try:
        dobj = load('config.toml')

        child = pexpect.spawn('ssh %s@kagayaki' % dobj['configure']['user'])
        child.expect("%s" % dobj['configure']['user'])
        child.sendline('%s' % dobj['configure']['pass'])
        child.expect("%s" % dobj['configure']['user'])

        print(child.before.decode(encoding='utf-8'))
        print(child.after.decode(encoding='utf-8'))

        child.interact()

    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
