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

        luser = dobj['kagayaki_configure']['lusr']
        cmptr = dobj['kagayaki_configure']['comp']
        uname = dobj['kagayaki_configure']['user']
        pword = dobj['kagayaki_configure']['pass']

        pstr = f'sshfs {uname}@{cmptr}:/home/{uname} /home/{luser}/{cmptr}'
        p = pexpect.spawn(pstr, ignore_sighup=True)

        p.expect("%s" % dobj['kagayaki_configure']['user'])
        p.sendline('%s' % dobj['kagayaki_configure']['pass'])
        #print(p.before.decode(encoding='utf-8'))
        #print(p.after.decode(encoding='utf-8'))
        p.expect(pexpect.EOF)

    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == '__main__':
    main()
