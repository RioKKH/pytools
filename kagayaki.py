#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import toml
import traceback
from typing import Dict

import pexpect

PYTOOLDIR="/home/kakehi/github/pytools"

def load(fin:str) -> Dict:
    with open(fin) as fr:
        dobj = toml.load(fr)
    return dobj

def main():
    try:
        dobj = load('/'.join([PYTOOLDIR, 'config.toml']))

        child = pexpect.spawn('ssh %s@kagayaki -Y' % dobj['kagayaki_configure']['user'])
        child.expect("%s" % dobj['kagayaki_configure']['user'])
        child.sendline('%s' % dobj['kagayaki_configure']['pass'])
        child.expect("%s" % dobj['kagayaki_configure']['user'])

        print(child.before.decode(encoding='utf-8'))
        print(child.after.decode(encoding='utf-8'))

        child.interact()

    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
